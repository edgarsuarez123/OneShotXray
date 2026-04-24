"""
analysis/roc.py -- Simulated ROC curve for hemorrhage detection (NIH Aim 2).

Strategy:
  - 10 lesion cases:    seeds 42-51, phantom_lesion_5mm.h5
  - 10 no-lesion cases: seeds 52-61, phantom_nolesion.h5
  - For each case: re-noise the pre-computed clean sinogram (efficient -- no new FP3D),
    re-run simulated centroiding + solver, reconstruct with mART, measure CNR at
    hemorrhage ROI.
  - Sweep threshold on CNR -> TP/FP rates -> ROC curve -> AUC

Sinogram re-use:
  - Lesion sinogram: already in sinogram_80_restricted.h5 (from run_nih.py)
  - No-lesion sinogram: generated here by forward-projecting phantom_nolesion.h5
    using the SAME cone_vec geometry as the lesion sinogram (ensures identical scanner
    geometry for fair comparison). Saved to sinogram_80_nolesion_restricted.h5.

Gate: AUC > 0.75

Output: data/nih/roc_results.h5
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forward.projector import forward_project, apply_noise_pipeline
from forward.centroiding import process_all_shots
from solver.solver import solve_shot
from solver.projection import params_to_cone_vec
from recon.mart import reconstruct_mart
from recon.inpainting import inpaint_markers
from analysis.metrics import compute_cnr, build_hemorrhage_masks


# ---- Parameters ---------------------------------------------------------------
N_LESION   = 10
N_NOLESION = 10
SEEDS_LES   = list(range(42, 42 + N_LESION))
SEEDS_NOLES = list(range(52, 52 + N_NOLESION))

I0          = 10_000
FOCAL_SPOT  = 0.5
SOD         = 500.0
ODD         = 200.0
DET_SPACING = 0.4
DET_ROWS    = 512
DET_COLS    = 512

NIH_GRID       = (200, 160, 140)
NIH_VOXEL_SIZE = 1.0
MART_ITER      = 25
MART_RELAX     = 0.5
CENTROID_NOISE_PX = 0.10

HEM_CENTER_MM = np.array([58.0, 20.0, 0.0])


def _generate_nolesion_sino(
    lesion_sino_path: Path,
    nolesion_phantom_path: Path,
    out_path: Path,
) -> None:
    """
    Forward-project no-lesion phantom using the exact same cone_vec geometry
    as the lesion sinogram (same shots, same perturbations). Saves clean + noisy
    sinogram so ROC trials can re-noise from the same expected fluence.
    """
    if out_path.exists():
        print(f'  No-lesion sinogram already exists: {out_path}')
        return

    print('  Generating no-lesion sinogram (same geometry as lesion)...')
    with h5py.File(lesion_sino_path, 'r') as f:
        vectors = f['shots/nominal/cone_vec'][:]

    with h5py.File(nolesion_phantom_path, 'r') as f:
        volume     = f['volume'][:]
        voxel_size = float(f.attrs['voxel_size_mm'])

    t0 = time.perf_counter()
    sino_clean = forward_project(volume, vectors, voxel_size, DET_ROWS, DET_COLS)
    # Apply one instance of noise for the "base" sinogram
    sino_noisy = apply_noise_pipeline(
        sino_clean, I0=I0, focal_spot=FOCAL_SPOT,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING, seed=99,
    )
    elapsed = time.perf_counter() - t0
    print(f'  Done in {elapsed:.1f}s')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('sinogram_clean', data=sino_clean, compression='gzip', compression_opts=4)
        f.create_dataset('sinogram',       data=sino_noisy, compression='gzip', compression_opts=4)


def _run_one_trial(
    sinogram_base: np.ndarray,
    gt_9dof: np.ndarray,
    vectors: np.ndarray,
    nominal_src: np.ndarray,
    marker_3d: np.ndarray,
    seed: int,
) -> float:
    """
    Run one ROC trial: re-noise + centroid + solver + mART reconstruct.
    Returns CNR at hemorrhage ROI.
    """
    # 1. Re-noise sinogram with new seed
    sino_noisy = apply_noise_pipeline(
        sinogram_base,
        I0=I0, focal_spot=FOCAL_SPOT,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING,
        seed=seed,
    )

    # 2. Simulated centroiding
    n_shots = sinogram_base.shape[1]
    centroid_result = process_all_shots(
        sino_noisy, gt_9dof, vectors,
        marker_3d.astype(np.float64),
        det_spacing=DET_SPACING,
        focal_spot_mm=FOCAL_SPOT,
        refine_window=8,
        simulated_noise_px=CENTROID_NOISE_PX,
        seed=seed,
    )
    positions      = centroid_result['positions']
    detection_mask = centroid_result['detection_mask']
    weights        = centroid_result['unsharpness_weights']

    # 3. Solve geometry
    recovered_9dof = np.full((n_shots, 9), np.nan)
    for i in range(n_shots):
        res = solve_shot(
            marker_3d, positions[i], weights[i], detection_mask[i],
            DET_SPACING, DET_ROWS, DET_COLS, SOD, ODD,
            nominal_src=nominal_src[i],
        )
        recovered_9dof[i] = res['params9']

    recovered_vecs = np.array([
        params_to_cone_vec(recovered_9dof[i], DET_SPACING)
        for i in range(n_shots)
    ])

    # 4. mART reconstruction
    vol, _ = reconstruct_mart(
        sino_noisy, recovered_vecs,
        n_iter=MART_ITER, relaxation=MART_RELAX,
        voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
    )
    vol = inpaint_markers(
        vol, marker_3d,
        marker_radius_mm=1.5, shell_outer_mm=4.5,
        voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
    )

    # 5. CNR at hemorrhage ROI
    defect_mask, bg_mask = build_hemorrhage_masks(
        hemorrhage_center_mm=HEM_CENTER_MM,
        lesion_radius_mm=5.0,
        grid_size=NIH_GRID,
        voxel_size=NIH_VOXEL_SIZE,
    )
    cnr = compute_cnr(vol, defect_mask, bg_mask)
    return float(cnr)


def compute_roc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple:
    """ROC curve + AUC (trapezoid)."""
    all_scores = np.concatenate([scores_pos, scores_neg])
    all_labels = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    thresholds = np.sort(np.unique(all_scores))[::-1]

    fprs, tprs = [], []
    for thr in thresholds:
        pred = all_scores >= thr
        tp = int(np.sum(pred & (all_labels == 1)))
        fp = int(np.sum(pred & (all_labels == 0)))
        fn = int(np.sum(~pred & (all_labels == 1)))
        tn = int(np.sum(~pred & (all_labels == 0)))
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))

    fpr = np.array([0.0] + fprs + [1.0])
    tpr = np.array([0.0] + tprs + [1.0])
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr))
    return fpr, tpr, auc


def main() -> None:
    data_dir         = ROOT / 'data' / 'nih'
    out_path         = data_dir / 'roc_results.h5'
    lesion_sino_path = data_dir / 'sinogram_80_restricted.h5'
    nolesion_sino_path = data_dir / 'sinogram_80_nolesion_restricted.h5'
    lesion_phantom   = data_dir / 'phantom_lesion_5mm.h5'
    nolesion_phantom = data_dir / 'phantom_nolesion.h5'

    print('=' * 60)
    print('NIH Aim 2 -- Simulated ROC')
    print(f'  Lesion trials    : {N_LESION} (seeds {SEEDS_LES[0]}-{SEEDS_LES[-1]})')
    print(f'  No-lesion trials : {N_NOLESION} (seeds {SEEDS_NOLES[0]}-{SEEDS_NOLES[-1]})')
    print('=' * 60)

    # Generate no-lesion sinogram with same geometry
    print('\n[1] Preparing no-lesion sinogram...')
    _generate_nolesion_sino(lesion_sino_path, nolesion_phantom, nolesion_sino_path)

    # Load shared geometry (same for both lesion + no-lesion)
    with h5py.File(lesion_sino_path, 'r') as f:
        sinogram_lesion = f['sinogram'][:]
        gt_9dof         = f['shots/nominal/ground_truth_9dof'][:]
        vectors         = f['shots/nominal/cone_vec'][:]
        nominal_src     = f['shots/nominal/source_positions'][:]

    with h5py.File(nolesion_sino_path, 'r') as f:
        sinogram_nolesion = f['sinogram'][:]

    with h5py.File(lesion_phantom, 'r') as f:
        marker_3d_les = f['marker_positions'][:].astype(np.float64)

    with h5py.File(nolesion_phantom, 'r') as f:
        marker_3d_noles = f['marker_positions'][:].astype(np.float64)

    cnr_lesion   = []
    cnr_nolesion = []

    # Lesion trials
    print(f'\n[2] Running {N_LESION} lesion trials...')
    for trial_idx, seed in enumerate(SEEDS_LES):
        t0 = time.perf_counter()
        print(f'  Lesion trial {trial_idx+1}/{N_LESION} (seed={seed}) ...', end='', flush=True)
        cnr = _run_one_trial(sinogram_lesion, gt_9dof, vectors, nominal_src, marker_3d_les, seed)
        cnr_lesion.append(cnr)
        print(f'  CNR={cnr:.3f}  ({time.perf_counter()-t0:.1f}s)')

    # No-lesion trials
    print(f'\n[3] Running {N_NOLESION} no-lesion trials...')
    for trial_idx, seed in enumerate(SEEDS_NOLES):
        t0 = time.perf_counter()
        print(f'  No-lesion trial {trial_idx+1}/{N_NOLESION} (seed={seed}) ...', end='', flush=True)
        cnr = _run_one_trial(sinogram_nolesion, gt_9dof, vectors, nominal_src, marker_3d_noles, seed)
        cnr_nolesion.append(cnr)
        print(f'  CNR={cnr:.3f}  ({time.perf_counter()-t0:.1f}s)')

    # ROC
    cnr_les_arr   = np.array(cnr_lesion)
    cnr_noles_arr = np.array(cnr_nolesion)
    fpr, tpr, auc = compute_roc(cnr_les_arr, cnr_noles_arr)

    print(f'\n=== Simulated ROC Results ===')
    print(f'  Lesion CNR   : mean={cnr_les_arr.mean():.3f}  '
          f'std={cnr_les_arr.std():.3f}  range=[{cnr_les_arr.min():.2f}, {cnr_les_arr.max():.2f}]')
    print(f'  No-lesion CNR: mean={cnr_noles_arr.mean():.3f}  '
          f'std={cnr_noles_arr.std():.3f}  range=[{cnr_noles_arr.min():.2f}, {cnr_noles_arr.max():.2f}]')
    print(f'  AUC          : {auc:.4f}  '
          f'({"PASS" if auc > 0.75 else "FAIL"} -- target > 0.75)')

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('cnr_lesion',   data=cnr_les_arr)
        f.create_dataset('cnr_nolesion', data=cnr_noles_arr)
        f.create_dataset('fpr',          data=fpr)
        f.create_dataset('tpr',          data=tpr)
        f.attrs['auc']             = auc
        f.attrs['n_lesion']        = N_LESION
        f.attrs['n_nolesion']      = N_NOLESION
        f.attrs['seeds_lesion']    = np.array(SEEDS_LES)
        f.attrs['seeds_nolesion']  = np.array(SEEDS_NOLES)
    print(f'\n  Saved -> {out_path}')

    if auc <= 0.75:
        print('\n*** ROC GATE FAIL: AUC <= 0.75 ***')
        sys.exit(1)


if __name__ == '__main__':
    main()
