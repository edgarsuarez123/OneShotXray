"""
solver/solver.py — Per-shot SDSG orchestration: anchor loop → merge → final residuals.
"""

import numpy as np

from solver.u7 import solve_u7
from solver.merge import merge_u7_solutions
from solver.projection import params_to_cone_vec, project_points_batch


def solve_shot(
    marker_3d: np.ndarray,
    observed_2d: np.ndarray,
    weights: np.ndarray,
    detection_mask: np.ndarray,
    det_spacing: float,
    det_rows: int,
    det_cols: int,
    sod: float,
    odd: float,
) -> dict:
    """
    Run N_markers U7 problems for one shot and merge into a single 9-DOF estimate.

    Parameters
    ----------
    marker_3d : (N_markers, 3) mm
    observed_2d : (N_markers, 2) [row, col] pixels — NaN where undetected
    weights : (N_markers,) unsharpness weights
    detection_mask : (N_markers,) bool
    det_spacing : float mm
    det_rows, det_cols : int
    sod, odd : float mm

    Returns
    -------
    dict with keys:
        params9              : (9,) merged geometry [src_xyz, det_xyz, euler_abc_deg]
        per_marker_residuals : (N_markers,) px, NaN for undetected
        per_shot_rms         : float px (NaN if < 4 markers detected)
        u7_costs             : (N_markers,) unweighted px RMS per anchor, NaN if skipped
        u7_failed_mask       : (N_markers,) bool — True if U7 failed or skipped
    """
    n_markers  = len(marker_3d)
    n_detected = int(detection_mask.sum())

    # Under-constrained: need >= 4 markers to reliably solve 9 DOF
    if n_detected < 4:
        fallback = np.array([0.0, 0.0, -sod, 0.0, 0.0, odd, 0.0, 0.0, 0.0])
        return {
            'params9':              fallback,
            'per_marker_residuals': np.full(n_markers, np.nan),
            'per_shot_rms':         np.nan,
            'u7_costs':             np.full(n_markers, np.nan),
            'u7_failed_mask':       np.ones(n_markers, dtype=bool),
        }

    # Run one U7 per anchor
    results = []
    for k in range(n_markers):
        if not detection_mask[k]:
            results.append({
                'params9': None,
                'cost_px': np.nan,
                'success': False,
                'nfev':    0,
                'message': 'anchor not detected',
                'anchor_idx': k,
            })
            continue

        res = solve_u7(
            marker_3d, observed_2d, weights,
            anchor_idx=k,
            det_spacing=det_spacing,
            det_rows=det_rows,
            det_cols=det_cols,
            sod=sod,
            odd=odd,
        )
        res['anchor_idx'] = k
        results.append(res)

    # Merge all valid U7 results into a single geometry estimate
    merged_params9 = merge_u7_solutions(results)

    # Compute final per-marker reprojection residuals using merged geometry
    cone_vec = params_to_cone_vec(merged_params9, det_spacing)
    pred = project_points_batch(
        marker_3d,
        cone_vec[0:3], cone_vec[3:6],
        cone_vec[6:9], cone_vec[9:12],
        det_rows, det_cols,
    )

    per_marker_residuals = np.full(n_markers, np.nan, dtype=np.float64)
    sq_errors = []
    for j in range(n_markers):
        obs = observed_2d[j]
        if np.any(np.isnan(obs)) or np.any(np.isnan(pred[j])):
            continue
        dr = pred[j, 0] - obs[0]
        dc = pred[j, 1] - obs[1]
        err = np.sqrt(dr**2 + dc**2)
        per_marker_residuals[j] = err
        sq_errors.append(err**2)

    per_shot_rms = float(np.sqrt(np.mean(sq_errors))) if sq_errors else np.nan

    u7_costs = np.array([
        r['cost_px'] if (r['cost_px'] is not None and not (r['params9'] is None))
        else np.nan
        for r in results
    ])
    u7_failed_mask = np.array([
        (r['params9'] is None)
        or (not r['success'])
        or (not np.isfinite(r['cost_px']))
        or (r['cost_px'] > 2.0)
        for r in results
    ], dtype=bool)

    return {
        'params9':              merged_params9,
        'per_marker_residuals': per_marker_residuals,
        'per_shot_rms':         per_shot_rms,
        'u7_costs':             u7_costs,
        'u7_failed_mask':       u7_failed_mask,
    }
