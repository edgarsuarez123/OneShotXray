"""
solver/run_nih_solver.py — SDSG solver pipeline for NIH cranial phantom.

Reads:  data/nih/sinogram_80_{arc}.h5  (centroids + GT geometry)
        (marker_positions stored inside sinogram file)
Writes: data/nih/sinogram_80_{arc}.h5  group 'results/'

Gate: mean residual < 0.3 px  (NIH gate, looser than Navy 0.2px due to σ_s=3mm)
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solver.solver import solve_shot
from solver.projection import params_to_cone_vec, euler_to_R


NIH_GATE_PX = 0.3


def geodesic_rotation_error(euler_est, euler_gt):
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


def run_nih_solver(arc: str = 'restricted') -> None:
    t0 = time.perf_counter()

    sino_path = ROOT / 'data' / 'nih' / f'sinogram_80_{arc}.h5'

    print('=' * 60)
    print(f'SDSG Solver — NIH {arc} arc')
    print('=' * 60)

    with h5py.File(sino_path, 'r') as f:
        positions      = f['centroids/positions'][:]
        detection_mask = f['centroids/detection_mask'][:].astype(bool)
        weights        = f['centroids/unsharpness_weights'][:]
        gt_9dof        = f['shots/nominal/ground_truth_9dof'][:]
        nominal_src    = f['shots/nominal/source_positions'][:]
        marker_3d      = f['marker_positions'][:].astype(np.float64)
        det_spacing    = float(f.attrs['det_spacing_mm'])
        det_rows       = int(f.attrs['det_rows'])
        det_cols       = int(f.attrs['det_cols'])
        sod            = float(f.attrs['sod_mm'])
        odd            = float(f.attrs['odd_mm'])

    n_shots, n_markers, _ = positions.shape
    print(f'\n[1/4] Loaded: {n_shots} shots, {n_markers} markers, '
          f'det={det_rows}×{det_cols}, spacing={det_spacing}mm')

    print(f'\n[2/4] Running SDSG solver...')
    recovered_9dof       = np.full((n_shots, 9), np.nan)
    per_marker_residuals = np.full((n_shots, n_markers), np.nan)
    per_shot_rms         = np.full(n_shots, np.nan)
    u7_costs             = np.full((n_shots, n_markers), np.nan)
    u7_failed_mask       = np.zeros((n_shots, n_markers), dtype=bool)

    t_solve = time.perf_counter()
    for i in tqdm(range(n_shots), desc='Solving', unit='shot'):
        res = solve_shot(
            marker_3d, positions[i], weights[i], detection_mask[i],
            det_spacing, det_rows, det_cols, sod, odd,
            nominal_src=nominal_src[i],
        )
        recovered_9dof[i]       = res['params9']
        per_marker_residuals[i] = res['per_marker_residuals']
        per_shot_rms[i]         = res['per_shot_rms']
        u7_costs[i]             = res['u7_costs']
        u7_failed_mask[i]       = res['u7_failed_mask']
    print(f'      Solver time: {time.perf_counter() - t_solve:.1f}s')

    print('\n[3/4] Computing accuracy metrics...')
    recovered_cone_vec = np.array([
        params_to_cone_vec(recovered_9dof[i], det_spacing)
        for i in range(n_shots)
    ])
    position_error_mm = np.linalg.norm(
        recovered_9dof[:, 0:3] - gt_9dof[:, 0:3], axis=1)
    angular_error_deg = geodesic_rotation_error(
        recovered_9dof[:, 6:9], gt_9dof[:, 6:9])

    mean_res   = float(np.nanmean(per_shot_rms))
    median_res = float(np.nanmedian(per_shot_rms))
    p95_res    = float(np.nanpercentile(per_shot_rms, 95))
    n_valid    = int(np.sum(~np.isnan(per_shot_rms)))

    print('\n[4/4] Writing results...')
    with h5py.File(sino_path, 'a') as f:
        if 'results' in f:
            del f['results']
        g = f.create_group('results')
        g.create_dataset('recovered_9dof',     data=recovered_9dof)
        g.create_dataset('recovered_cone_vec', data=recovered_cone_vec)
        g.create_dataset('u7_costs',           data=u7_costs)
        g.create_dataset('u7_failed_mask',     data=u7_failed_mask)
        g.create_dataset('position_error_mm',  data=position_error_mm)
        g.create_dataset('angular_error_deg',  data=angular_error_deg)
        rg = g.create_group('residuals')
        rg.create_dataset('per_marker', data=per_marker_residuals)
        rg.create_dataset('per_shot',   data=per_shot_rms)
        rg.attrs['mean']   = mean_res
        rg.attrs['median'] = median_res
        rg.attrs['p95']    = p95_res
        g.attrs['wall_time_s'] = time.perf_counter() - t0

    wall = time.perf_counter() - t0
    status = 'PASS' if mean_res < NIH_GATE_PX else 'FAIL'
    print(f'\n{"=" * 60}')
    print(f'NIH Solver ({arc}) done in {wall:.1f}s')
    print(f'  Mean residual   : {mean_res:.4f} px  ({status} — gate < {NIH_GATE_PX}px)')
    print(f'  Median residual : {median_res:.4f} px')
    print(f'  P95 residual    : {p95_res:.4f} px')
    print(f'  Shots solved    : {n_valid}/{n_shots}')
    print(f'  Mean pos error  : {np.nanmean(position_error_mm):.3f} mm')
    print(f'  Mean ang error  : {np.nanmean(angular_error_deg):.3f} deg')
    print('=' * 60)

    if mean_res >= NIH_GATE_PX:
        print(f'\n*** GATE FAIL: mean residual {mean_res:.4f} >= {NIH_GATE_PX} ***')
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--arc', choices=['restricted', 'full360', 'both'], default='both')
    args = p.parse_args()
    arcs = ['restricted', 'full360'] if args.arc == 'both' else [args.arc]
    for arc in arcs:
        run_nih_solver(arc)
