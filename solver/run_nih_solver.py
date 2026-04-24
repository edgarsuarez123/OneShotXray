"""
solver/run_nih_solver.py — NIH SDSG solver pipeline.

Reads:  data/nih/sinogram_80_restricted.h5
        data/nih/sinogram_80_full360.h5
        data/nih/phantom_lesion_5mm.h5
Writes: solver results/  group back into the sinogram HDF5 files

Gate: mean residual < 0.3px (NIH-SHOT-006, more lenient than Navy 0.2px)
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


def solve_sinogram(sino_path: Path, phantom_path: Path, gate_px: float = 0.3) -> dict:
    """Run SDSG solver on one sinogram file, write results, return summary."""
    t0 = time.perf_counter()
    arc_label = sino_path.stem

    print(f'\n{"=" * 60}')
    print(f'SDSG Solver — NIH  [{arc_label}]')
    print(f'{"=" * 60}')

    # ── Load ──────────────────────────────────────────────────────
    print(f'\n[1/4] Loading from {sino_path}')
    with h5py.File(sino_path, 'r') as f:
        positions      = f['centroids/positions'][:]
        detection_mask = f['centroids/detection_mask'][:].astype(bool)
        weights        = f['centroids/unsharpness_weights'][:]
        gt_9dof        = f['shots/nominal/ground_truth_9dof'][:]
        nominal_src    = f['shots/nominal/source_positions'][:]
        det_spacing    = float(f.attrs['det_spacing_mm'])
        det_rows       = int(f.attrs['det_rows'])
        det_cols       = int(f.attrs['det_cols'])
        sod            = float(f.attrs['sod_mm'])
        odd            = float(f.attrs['odd_mm'])

    with h5py.File(phantom_path, 'r') as f:
        marker_3d = f['marker_positions'][:].astype(np.float64)

    n_shots, n_markers, _ = positions.shape
    print(f'      shots={n_shots}, markers={n_markers}, det_spacing={det_spacing}mm')

    # ── Solve ─────────────────────────────────────────────────────
    print(f'\n[2/4] Running solver ({n_shots} shots × {n_markers} anchors each)...')
    recovered_9dof       = np.full((n_shots, 9), np.nan)
    per_marker_residuals = np.full((n_shots, n_markers), np.nan)
    per_shot_rms         = np.full(n_shots, np.nan)
    u7_costs             = np.full((n_shots, n_markers), np.nan)
    u7_failed_mask       = np.zeros((n_shots, n_markers), dtype=bool)

    t_solve = time.perf_counter()
    for i in tqdm(range(n_shots), desc='Solving shots', unit='shot'):
        res = solve_shot(
            marker_3d,
            positions[i],
            weights[i],
            detection_mask[i],
            det_spacing, det_rows, det_cols, sod, odd,
            nominal_src=nominal_src[i],
        )
        recovered_9dof[i]       = res['params9']
        per_marker_residuals[i] = res['per_marker_residuals']
        per_shot_rms[i]         = res['per_shot_rms']
        u7_costs[i]             = res['u7_costs']
        u7_failed_mask[i]       = res['u7_failed_mask']
    t_solve = time.perf_counter() - t_solve
    print(f'      Solver wall time: {t_solve:.1f}s')

    # ── Metrics ───────────────────────────────────────────────────
    print('\n[3/4] Computing accuracy vs GT...')
    recovered_cone_vec = np.array([
        params_to_cone_vec(recovered_9dof[i], det_spacing)
        for i in range(n_shots)
    ])
    position_error_mm = np.linalg.norm(
        recovered_9dof[:, 0:3] - gt_9dof[:, 0:3], axis=1
    )
    angular_error_deg = geodesic_rotation_error(
        recovered_9dof[:, 6:9], gt_9dof[:, 6:9]
    )
    mean_res   = float(np.nanmean(per_shot_rms))
    median_res = float(np.nanmedian(per_shot_rms))
    p95_res    = float(np.nanpercentile(per_shot_rms, 95))
    n_valid    = int(np.sum(~np.isnan(per_shot_rms)))

    # ── Write results ─────────────────────────────────────────────
    print(f'\n[4/4] Writing results to {sino_path}...')
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

    # ── Summary ───────────────────────────────────────────────────
    wall = time.perf_counter() - t0
    gate_pass = mean_res < gate_px
    print(f'\n{"=" * 60}')
    print(f'Solver complete in {wall:.1f}s  [{arc_label}]')
    print(f'  Mean residual : {mean_res:.4f} px  '
          f'({"PASS" if gate_pass else "FAIL"} — target < {gate_px} px)')
    print(f'  Median residual: {median_res:.4f} px')
    print(f'  P95 residual  : {p95_res:.4f} px')
    print(f'  Mean pos error: {np.nanmean(position_error_mm):.3f} mm')
    print(f'  Mean ang error: {np.nanmean(angular_error_deg):.3f} deg')
    print(f'  Shots solved  : {n_valid}/{n_shots}')
    print('=' * 60)

    return {
        'arc': arc_label,
        'mean_res': mean_res,
        'gate_pass': gate_pass,
        'n_valid': n_valid,
        'n_shots': n_shots,
    }


def main() -> None:
    phantom_path = ROOT / 'data' / 'nih' / 'phantom_lesion_5mm.h5'
    NIH_GATE_PX  = 0.3

    results = []

    # Restricted 180° arc (primary — gate check here)
    r1 = solve_sinogram(
        ROOT / 'data' / 'nih' / 'sinogram_80_restricted.h5',
        phantom_path,
        gate_px=NIH_GATE_PX,
    )
    results.append(r1)

    # Full 360° arc (comparison)
    r2 = solve_sinogram(
        ROOT / 'data' / 'nih' / 'sinogram_80_full360.h5',
        phantom_path,
        gate_px=NIH_GATE_PX,
    )
    results.append(r2)

    print('\n=== NIH Solver Summary ===')
    for r in results:
        status = 'PASS' if r['gate_pass'] else 'FAIL'
        print(f"  {r['arc']:30s}  residual={r['mean_res']:.4f}px  [{status}]")

    # Gate exit code for restricted arc
    if not r1['gate_pass']:
        print('\n*** NIH GATE FAIL: restricted arc mean residual >= 0.3 px ***')
        sys.exit(1)


if __name__ == '__main__':
    main()
