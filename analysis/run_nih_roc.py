"""
analysis/run_nih_roc.py -- Scaled ROC analysis (n=200: 100 lesion + 100 no-lesion).

Strategy:
  1. For each trial: draw a fresh perturbed geometry (geometry_seed = 20000+k).
  2. Forward-project the clean phantom through that geometry.
  3. Re-apply Poisson noise (poisson_seed varies per trial).
  4. Run SDSG solver + mART 25 iter lam=0.5 -> measure CNR at hemorrhage ROI.
  5. Save incrementally every trial -- crash mid-run loses <=1 trial.
  6. Use --resume to skip already-completed trials.

Geometry seeds (20000+k for lesion, 20100+k for no-lesion) are disjoint from
Poisson seeds (42-241) and centroid seeds (+10000).

AUC: non-parametric (Mann-Whitney), no sklearn dependency.
95% CI: Hanley-McNeil formula (analytical).

Gates:
  AUC > 0.75            (original Phase I target)
  AUC CI lower > 0.85   (new gate -- n=200 drops SE from ~0.060 to ~0.019)
"""

import sys
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forward.geometry import perturb_geometry, geometry_to_cone_vec
from forward.projector import forward_project, apply_noise_pipeline
from forward.nih_centroiding import simulate_centroids
from forward.nih_geometry import (
    generate_restricted_arc_shots,
    SOD, ODD, SIGMA_S, SIGMA_THETA, N_SHOTS,
    DET_ROWS, DET_COLS, DET_SPACING, I0,
)
from phantom.nih_phantom import NX, NY, NZ, VOXEL_SIZE
from solver.solver import solve_shot
from solver.projection import params_to_cone_vec
from recon.mart import reconstruct_mart
from analysis.metrics import compute_cnr, build_nih_hemorrhage_mask, build_nih_bg_mask

# Trial counts
N_LESION    = 100   # Poisson seeds 42-141
N_NOLESION  = 100   # Poisson seeds 142-241
LESION_SEED_OFFSET    = 42
NOLESION_SEED_OFFSET  = 142

# Geometry seeds: disjoint from Poisson (42-241) and centroid (+10000)
GEOMETRY_SEED_OFFSET = 20_000   # lesion: 20000-20099, no-lesion: 20100-20199

# mART parameters (optimal from Aim 1 sweep)
MART_N_ITER = 25
MART_LAM    = 0.5

# Gates
AUC_GATE      = 0.75
AUC_CI_GATE   = 0.85


# ---------------------------------------------------------------------------
# AUC / CI utilities (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

def compute_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Non-parametric (Mann-Whitney) AUC estimate."""
    n_pos, n_neg = len(pos), len(neg)
    greater = np.sum(pos[:, None] > neg[None, :])
    tied    = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * tied) / (n_pos * n_neg))


def hanley_mcneil_ci(auc: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> tuple:
    """
    Approximate 95% CI using the Hanley-McNeil (1982) formula.

    Q1 = AUC / (2 - AUC)
    Q2 = 2 * AUC^2 / (1 + AUC)
    SE = sqrt((AUC(1-AUC) + (n_pos-1)(Q1-AUC^2) + (n_neg-1)(Q2-AUC^2)) / (n_pos*n_neg))
    CI = AUC +/- z * SE
    """
    from scipy.stats import norm
    Q1 = auc / (2.0 - auc)
    Q2 = 2.0 * auc**2 / (1.0 + auc)
    SE = np.sqrt(
        (auc * (1 - auc)
         + (n_pos - 1) * (Q1 - auc**2)
         + (n_neg - 1) * (Q2 - auc**2))
        / (n_pos * n_neg)
    )
    z = norm.ppf(1.0 - alpha / 2.0)
    return float(auc - z * SE), float(auc + z * SE), float(SE)


# ---------------------------------------------------------------------------
# Single-trial pipeline
# ---------------------------------------------------------------------------

def run_trial(
    vol: np.ndarray,
    nominal_src: np.ndarray,
    marker_3d: np.ndarray,
    geometry_seed: int,
    poisson_seed: int,
) -> float:
    """
    Run one full pipeline trial and return CNR at hemorrhage ROI.

    Each trial draws a fresh perturbed geometry so trial-to-trial variation
    reflects realistic freehand scan-to-scan differences (sigma_s=3mm, sigma_theta=1.5deg).

    Parameters
    ----------
    vol           : (NX, NY, NZ) float32 phantom volume
    nominal_src   : (N_shots, 3) deterministic Fibonacci arc source positions
    marker_3d     : (N_markers, 3) mm
    geometry_seed : int -- unique per trial; controls perturb_geometry draw
    poisson_seed  : int -- unique per trial; controls Poisson and centroid noise

    Returns
    -------
    cnr : float  (NaN if solver fails on >= half the shots)
    """
    # 1. Per-trial ground-truth geometry
    gt_9dof    = perturb_geometry(nominal_src, SOD, ODD, SIGMA_S, SIGMA_THETA,
                                  seed=geometry_seed)
    vectors_gt = geometry_to_cone_vec(gt_9dof, DET_SPACING, DET_ROWS, DET_COLS)

    # 2. Forward project through this trial's geometry
    clean_sino = forward_project(vol, vectors_gt, VOXEL_SIZE, DET_ROWS, DET_COLS)

    # 3. Apply Poisson noise (seed varies per trial)
    sino_noisy = apply_noise_pipeline(
        clean_sino, I0=I0, focal_spot=0.5,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING,
        seed=poisson_seed,
    )

    # 4. CRB centroid noise (seed offset to keep independent from Poisson noise)
    positions, detection_mask, weights = simulate_centroids(
        gt_9dof, marker_3d, DET_SPACING, DET_ROWS, DET_COLS,
        seed=poisson_seed + 10_000,
    )

    # 5. SDSG solver
    recovered_9dof = np.full((N_SHOTS, 9), np.nan)
    per_shot_rms   = np.full(N_SHOTS, np.nan)

    for i in range(N_SHOTS):
        res = solve_shot(
            marker_3d, positions[i], weights[i], detection_mask[i],
            DET_SPACING, DET_ROWS, DET_COLS, SOD, ODD,
            nominal_src=nominal_src[i],
        )
        recovered_9dof[i] = res['params9']
        per_shot_rms[i]   = res['per_shot_rms']

    n_solved = int(np.sum(~np.isnan(per_shot_rms)))
    if n_solved < N_SHOTS // 2:
        return np.nan   # too many failures -> exclude trial

    recovered_vecs = np.array([
        params_to_cone_vec(recovered_9dof[i], DET_SPACING)
        for i in range(N_SHOTS)
    ])

    # 6. mART reconstruction
    vol_out, _ = reconstruct_mart(
        sino_noisy, recovered_vecs,
        n_iter=MART_N_ITER, lam=MART_LAM,
        voxel_size=VOXEL_SIZE, grid_nx=NX, grid_ny=NY, grid_nz=NZ,
    )

    # 7. CNR at hemorrhage ROI
    hem_mask = build_nih_hemorrhage_mask()
    bg_mask  = build_nih_bg_mask()
    return compute_cnr(vol_out, hem_mask, bg_mask)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--resume', action='store_true',
                   help='Skip already-completed trials in the HDF5 file')
    args = p.parse_args()

    t0 = time.perf_counter()
    out_path = ROOT / 'data' / 'nih' / 'roc_results.h5'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'NIH ROC Analysis  n={N_LESION + N_NOLESION}  '
          f'({N_LESION} lesion + {N_NOLESION} no-lesion)')
    print('  geometry_reseeded=True (per-trial perturb_geometry)')
    print('=' * 60)

    # Initialise or load existing results
    cnr_lesion   = np.full(N_LESION,   np.nan)
    cnr_nolesion = np.full(N_NOLESION, np.nan)

    if args.resume and out_path.exists():
        with h5py.File(out_path, 'r') as f:
            if 'cnr_lesion' in f:
                saved = f['cnr_lesion'][:]
                cnr_lesion[:len(saved)] = saved[:N_LESION]
            if 'cnr_nolesion' in f:
                saved = f['cnr_nolesion'][:]
                cnr_nolesion[:len(saved)] = saved[:N_NOLESION]
        n_done_l  = int(np.sum(~np.isnan(cnr_lesion)))
        n_done_nl = int(np.sum(~np.isnan(cnr_nolesion)))
        print(f'Resuming: {n_done_l}/{N_LESION} lesion, '
              f'{n_done_nl}/{N_NOLESION} no-lesion already done')

    # Load phantoms
    phantom_dir = ROOT / 'data' / 'nih'

    print('\n[Setup] Loading phantoms...')
    with h5py.File(phantom_dir / 'phantom_lesion_5mm.h5', 'r') as f:
        vol_lesion   = f['volume'][:]
        marker_3d    = f['marker_positions'][:].astype(np.float64)

    with h5py.File(phantom_dir / 'phantom_nolesion.h5', 'r') as f:
        vol_nolesion = f['volume'][:]

    # Nominal arc (deterministic -- same for all trials; only the perturbation re-seeds)
    nominal_src = generate_restricted_arc_shots(N_SHOTS, SOD)
    print('[Setup] Done. Starting trial loop...\n')

    # Run lesion trials
    print(f'--- Lesion trials (geom seeds {GEOMETRY_SEED_OFFSET}-'
          f'{GEOMETRY_SEED_OFFSET + N_LESION - 1}, '
          f'poisson seeds {LESION_SEED_OFFSET}-{LESION_SEED_OFFSET + N_LESION - 1}) ---')
    for k in tqdm(range(N_LESION), desc='Lesion', unit='trial'):
        if not np.isnan(cnr_lesion[k]):
            continue   # already done (resume mode)
        cnr_lesion[k] = run_trial(
            vol_lesion, nominal_src, marker_3d,
            geometry_seed=GEOMETRY_SEED_OFFSET + k,
            poisson_seed=LESION_SEED_OFFSET + k,
        )

        # Incremental save after every trial
        with h5py.File(out_path, 'a') as f:
            if 'cnr_lesion' in f:
                del f['cnr_lesion']
            f.create_dataset('cnr_lesion', data=cnr_lesion)
            f.attrs['n_lesion']           = N_LESION
            f.attrs['n_nolesion']         = N_NOLESION
            f.attrs['geometry_reseeded']  = True

    # Run no-lesion trials
    print(f'\n--- No-lesion trials (geom seeds {GEOMETRY_SEED_OFFSET + N_LESION}-'
          f'{GEOMETRY_SEED_OFFSET + N_LESION + N_NOLESION - 1}, '
          f'poisson seeds {NOLESION_SEED_OFFSET}-{NOLESION_SEED_OFFSET + N_NOLESION - 1}) ---')
    for k in tqdm(range(N_NOLESION), desc='No-lesion', unit='trial'):
        if not np.isnan(cnr_nolesion[k]):
            continue
        cnr_nolesion[k] = run_trial(
            vol_nolesion, nominal_src, marker_3d,
            geometry_seed=GEOMETRY_SEED_OFFSET + N_LESION + k,
            poisson_seed=NOLESION_SEED_OFFSET + k,
        )

        with h5py.File(out_path, 'a') as f:
            if 'cnr_nolesion' in f:
                del f['cnr_nolesion']
            f.create_dataset('cnr_nolesion', data=cnr_nolesion)
            f.attrs['geometry_reseeded'] = True

    # Compute AUC + CI
    cnr_l_valid  = cnr_lesion[~np.isnan(cnr_lesion)]
    cnr_nl_valid = cnr_nolesion[~np.isnan(cnr_nolesion)]
    n_l, n_nl    = len(cnr_l_valid), len(cnr_nl_valid)

    auc            = compute_auc(cnr_l_valid, cnr_nl_valid)
    ci_lo, ci_hi, se = hanley_mcneil_ci(auc, n_l, n_nl)

    # Save final results + statistics
    with h5py.File(out_path, 'a') as f:
        f.attrs['auc']                = auc
        f.attrs['auc_ci_lo']          = ci_lo
        f.attrs['auc_ci_hi']          = ci_hi
        f.attrs['auc_se']             = se
        f.attrs['n_lesion_valid']     = n_l
        f.attrs['n_nolesion_valid']   = n_nl
        f.attrs['geometry_reseeded']  = True

    wall = time.perf_counter() - t0
    gate1 = 'PASS' if auc > AUC_GATE    else 'FAIL'
    gate2 = 'PASS' if ci_lo > AUC_CI_GATE else 'FAIL'

    print(f'\n{"=" * 60}')
    print(f'ROC Analysis complete in {wall/60:.1f} min')
    print(f'  n_lesion valid  : {n_l}/{N_LESION}')
    print(f'  n_nolesion valid: {n_nl}/{N_NOLESION}')
    print(f'  Lesion CNR      : mean={cnr_l_valid.mean():.3f}  std={cnr_l_valid.std():.3f}')
    print(f'  No-lesion CNR   : mean={cnr_nl_valid.mean():.3f}  std={cnr_nl_valid.std():.3f}')
    print(f'  AUC             : {auc:.4f}  SE={se:.4f}')
    print(f'  95% CI          : [{ci_lo:.4f}, {ci_hi:.4f}]')
    print(f'  GATE 1 (AUC > {AUC_GATE})        : {gate1}')
    print(f'  GATE 2 (CI_lo > {AUC_CI_GATE})   : {gate2}')
    print(f'  Results saved -> {out_path}')
    print('=' * 60)

    if gate1 == 'FAIL' or gate2 == 'FAIL':
        sys.exit(1)


if __name__ == '__main__':
    main()
