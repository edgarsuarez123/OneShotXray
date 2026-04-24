"""
forward/run_navy_variants.py — Shot-count variants + stress test pipeline.

Runs the full pipeline (forward → centroid → solver) for:
  - N = 20, 50, 200 shots (nominal: sigma_s=2mm, sigma_theta=1deg, seed=42)
  - N = 100 shots, stress: sigma_s=10mm, sigma_theta=5deg, seed=42

Outputs:
  data/navy/sinogram_20.h5
  data/navy/sinogram_50.h5
  data/navy/sinogram_200.h5
  data/navy/sinogram_100_stress.h5

Each HDF5 follows the same layout as sinogram_100.h5.
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forward.geometry    import generate_hemisphere_shots, perturb_geometry, geometry_to_cone_vec
from forward.projector   import forward_project, apply_noise_pipeline
from forward.centroiding import process_all_shots
from solver.solver       import solve_shot
from solver.projection   import params_to_cone_vec, euler_to_R

# ── Fixed geometry parameters ────────────────────────────────────────────────
SOD         = 500.0
ODD         = 200.0
DET_ROWS    = 512
DET_COLS    = 512
DET_SPACING = 0.2    # mm
FOCAL_SPOT  = 0.5    # mm
I0          = 10_000
SEED        = 42


def geodesic_rotation_error(euler_est: np.ndarray, euler_gt: np.ndarray) -> np.ndarray:
    n = len(euler_est)
    errors = np.full(n, np.nan)
    for i in range(n):
        if np.any(np.isnan(euler_est[i])) or np.any(np.isnan(euler_gt[i])):
            continue
        R_est = euler_to_R(euler_est[i])
        R_gt  = euler_to_R(euler_gt[i])
        R_err = R_gt.T @ R_est
        trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        errors[i] = np.degrees(np.arccos(trace))
    return errors


def run_variant(
    volume: np.ndarray,
    marker_3d: np.ndarray,
    voxel_size: float,
    n_shots: int,
    sigma_s: float,
    sigma_theta: float,
    label: str,
    out_path: Path,
) -> dict:
    """
    Run full pipeline for one variant and save to HDF5.

    Returns summary dict with mean/std residual, position error, angular error.
    """
    print(f'\n{"=" * 60}')
    print(f'Variant: {label}  (N={n_shots}, σ_s={sigma_s}mm, σ_θ={sigma_theta}°)')
    print('=' * 60)
    t0 = time.perf_counter()

    # Step 1: Generate shots
    source_positions = generate_hemisphere_shots(n_shots, SOD, seed=SEED)

    # Step 2: Perturb geometry
    gt_9dof = perturb_geometry(source_positions, SOD, ODD,
                               sigma_s=sigma_s, sigma_theta=sigma_theta, seed=SEED)
    vectors = geometry_to_cone_vec(gt_9dof, DET_SPACING, DET_ROWS, DET_COLS)

    # Step 3: Forward project
    print(f'  [1/4] Forward projection ({n_shots} shots)...')
    t_fp = time.perf_counter()
    sino_clean = forward_project(volume, vectors, voxel_size, DET_ROWS, DET_COLS)
    print(f'        Done in {time.perf_counter()-t_fp:.1f}s')

    # Step 4: Noise pipeline
    sino_noisy = apply_noise_pipeline(
        sino_clean, I0=I0, focal_spot=FOCAL_SPOT,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING, seed=SEED,
    )

    # Step 5: Centroiding
    print(f'  [2/4] Centroiding...')
    t_cent = time.perf_counter()
    centroid_result = process_all_shots(
        sino_noisy, gt_9dof, vectors,
        marker_3d.astype(np.float64),
        det_spacing=DET_SPACING,
        focal_spot_mm=FOCAL_SPOT,
    )
    print(f'        Done in {time.perf_counter()-t_cent:.1f}s')

    # Step 6: Solver
    print(f'  [3/4] SDSG solver ({n_shots} shots)...')
    t_solve = time.perf_counter()

    positions      = centroid_result['positions']      # (N, 8, 2)
    detection_mask = centroid_result['detection_mask'].astype(bool)
    weights        = centroid_result['unsharpness_weights']

    n_markers = len(marker_3d)
    recovered_9dof       = np.full((n_shots, 9),        np.nan)
    per_marker_residuals = np.full((n_shots, n_markers), np.nan)
    per_shot_rms         = np.full(n_shots,              np.nan)
    u7_costs             = np.full((n_shots, n_markers), np.nan)
    u7_failed_mask       = np.zeros((n_shots, n_markers), dtype=bool)

    for i in tqdm(range(n_shots), desc=f'  Solving {label}', unit='shot'):
        res = solve_shot(
            marker_3d.astype(np.float64),
            positions[i],
            weights[i],
            detection_mask[i],
            DET_SPACING, DET_ROWS, DET_COLS, SOD, ODD,
            nominal_src=source_positions[i],
        )
        recovered_9dof[i]       = res['params9']
        per_marker_residuals[i] = res['per_marker_residuals']
        per_shot_rms[i]         = res['per_shot_rms']
        u7_costs[i]             = res['u7_costs']
        u7_failed_mask[i]       = res['u7_failed_mask']

    print(f'        Done in {time.perf_counter()-t_solve:.1f}s')

    # Compute accuracy vs GT
    recovered_cone_vec = np.array([
        params_to_cone_vec(recovered_9dof[i], DET_SPACING)
        for i in range(n_shots)
    ])
    position_error_mm = np.linalg.norm(
        recovered_9dof[:, 0:3] - gt_9dof[:, 0:3], axis=1
    )
    angular_error_deg = geodesic_rotation_error(
        recovered_9dof[:, 6:9], gt_9dof[:, 6:9]
    )

    valid = per_shot_rms[~np.isnan(per_shot_rms)]
    mean_res = float(np.mean(valid)) if len(valid) else np.nan
    std_res  = float(np.std(valid))  if len(valid) else np.nan
    p95_res  = float(np.percentile(valid, 95)) if len(valid) else np.nan

    print(f'        Mean residual: {mean_res:.4f}px  '
          f'std: {std_res:.4f}px  P95: {p95_res:.4f}px')
    print(f'        Shots solved: {len(valid)}/{n_shots}')

    # Step 7: Save to HDF5
    print(f'  [4/4] Saving to {out_path}...')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('sinogram', data=sino_noisy,
                         compression='gzip', compression_opts=4)
        f['sinogram'].attrs['axis_order'] = '(det_rows, N_shots, det_cols)'

        g = f.create_group('shots/nominal')
        g.create_dataset('ground_truth_9dof',  data=gt_9dof)
        g.create_dataset('source_positions',   data=source_positions)
        g.create_dataset('cone_vec',           data=vectors)

        cg = f.create_group('centroids')
        cg.create_dataset('positions',           data=centroid_result['positions'])
        cg.create_dataset('ground_truth_2d',     data=centroid_result['ground_truth_2d'])
        cg.create_dataset('noise_per_shot',      data=centroid_result['noise_per_shot'])
        cg.create_dataset('unsharpness_weights', data=centroid_result['unsharpness_weights'])
        cg.create_dataset('detection_mask',      data=centroid_result['detection_mask'])

        rg = f.create_group('results')
        rg.create_dataset('recovered_9dof',     data=recovered_9dof)
        rg.create_dataset('recovered_cone_vec', data=recovered_cone_vec)
        rg.create_dataset('position_error_mm',  data=position_error_mm)
        rg.create_dataset('angular_error_deg',  data=angular_error_deg)
        rg.create_dataset('u7_costs',           data=u7_costs)
        rg.create_dataset('u7_failed_mask',     data=u7_failed_mask)

        residg = rg.create_group('residuals')
        residg.create_dataset('per_marker', data=per_marker_residuals)
        residg.create_dataset('per_shot',   data=per_shot_rms)
        residg.attrs['mean']   = mean_res
        residg.attrs['std']    = std_res
        residg.attrs['p95']    = p95_res

        f.attrs.update({
            'n_shots':         n_shots,
            'sod_mm':          SOD,
            'odd_mm':          ODD,
            'det_rows':        DET_ROWS,
            'det_cols':        DET_COLS,
            'det_spacing_mm':  DET_SPACING,
            'I0':              I0,
            'focal_spot_mm':   FOCAL_SPOT,
            'sigma_s_mm':      sigma_s,
            'sigma_theta_deg': sigma_theta,
            'seed':            SEED,
            'voxel_size_mm':   voxel_size,
            'variant_label':   label,
        })

    size_mb = out_path.stat().st_size / 1e6
    print(f'        Saved {size_mb:.1f} MB → {out_path.name}')
    print(f'        Total time: {time.perf_counter()-t0:.1f}s')

    return {
        'label':       label,
        'n_shots':     n_shots,
        'mean_res':    mean_res,
        'std_res':     std_res,
        'p95_res':     p95_res,
        'mean_pos_err': float(np.nanmean(position_error_mm)),
        'mean_ang_err': float(np.nanmean(angular_error_deg)),
        'shots_solved': int(len(valid)),
    }


def main() -> None:
    t_start = time.perf_counter()
    print('=' * 60)
    print('Navy shot-count variants + stress test')
    print('=' * 60)

    # Load phantom once
    phantom_path = ROOT / 'data' / 'navy' / 'phantom.h5'
    print(f'\nLoading phantom from {phantom_path}...')
    with h5py.File(phantom_path, 'r') as f:
        volume     = f['volume'][:].astype(np.float32)
        marker_3d  = f['marker_positions'][:].astype(np.float64)
        voxel_size = float(f.attrs['voxel_size_mm'])

    data_dir = ROOT / 'data' / 'navy'

    # Define variants: (n_shots, sigma_s, sigma_theta, label, filename)
    variants = [
        (20,  2.0, 1.0, 'N20_nominal',  data_dir / 'sinogram_20.h5'),
        (50,  2.0, 1.0, 'N50_nominal',  data_dir / 'sinogram_50.h5'),
        (200, 2.0, 1.0, 'N200_nominal', data_dir / 'sinogram_200.h5'),
        (100, 10.0, 5.0, 'N100_stress', data_dir / 'sinogram_100_stress.h5'),
    ]

    summaries = []
    for n_shots, sigma_s, sigma_theta, label, out_path in variants:
        if out_path.exists():
            print(f'\nSkipping {label} — {out_path.name} already exists')
            # Load summary from existing file
            with h5py.File(out_path, 'r') as f:
                if 'results/residuals/mean' in f or 'results/residuals' in f:
                    try:
                        per_shot = f['results/residuals/per_shot'][:]
                        pos_err  = f['results/position_error_mm'][:]
                        ang_err  = f['results/angular_error_deg'][:]
                        valid    = per_shot[~np.isnan(per_shot)]
                        summaries.append({
                            'label': label, 'n_shots': n_shots,
                            'mean_res': float(np.mean(valid)) if len(valid) else np.nan,
                            'std_res':  float(np.std(valid))  if len(valid) else 0.0,
                            'p95_res':  float(np.percentile(valid, 95)) if len(valid) else np.nan,
                            'mean_pos_err': float(np.nanmean(pos_err)),
                            'mean_ang_err': float(np.nanmean(ang_err)),
                            'shots_solved': int(len(valid)),
                        })
                    except Exception:
                        pass
            continue

        summary = run_variant(
            volume, marker_3d, voxel_size,
            n_shots, sigma_s, sigma_theta,
            label, out_path,
        )
        summaries.append(summary)

    # Print summary table
    print(f'\n{"=" * 60}')
    print(f'All variants complete in {time.perf_counter()-t_start:.1f}s')
    print(f'\n{"Label":<20} {"N":>5} {"Mean(px)":>10} {"Std(px)":>9} {"P95(px)":>9} '
          f'{"PosErr(mm)":>11} {"AngErr(deg)":>12} {"Solved":>7}')
    print('-' * 87)
    for s in summaries:
        print(f'{s["label"]:<20} {s["n_shots"]:>5} {s["mean_res"]:>10.4f} '
              f'{s["std_res"]:>9.4f} {s["p95_res"]:>9.4f} '
              f'{s["mean_pos_err"]:>11.2f} {s["mean_ang_err"]:>12.3f} '
              f'{s["shots_solved"]:>5}/{s["n_shots"]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
