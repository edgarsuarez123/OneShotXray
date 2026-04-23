"""
solver/u7.py — U7 cost function and single-anchor Levenberg-Marquardt solver.

The "U7" name comes from using 7 non-anchor markers as primary cost terms.
The anchor marker is included with 100x weight to emulate the patent's analytic
2-DOF constraint while remaining fully differentiable.

DECISION: rather than analytic elimination of 2 DOF (patent description), we keep 9
params free and weight the anchor residual 100x. Produces identical cost-minimum
geometry under nominal perturbation; avoids singularity when the anchor is near a
degenerate ray direction. See SOLVER_PLAN.md §4.
"""

import numpy as np
from scipy.optimize import least_squares

from solver.projection import params_to_cone_vec, project_points_batch


def u7_residuals(
    params9: np.ndarray,
    marker_3d: np.ndarray,
    observed_2d: np.ndarray,
    weights: np.ndarray,
    anchor_idx: int,
    det_spacing: float,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    Weighted residual vector for one U7 problem.

    Parameters
    ----------
    params9 : (9,) — [src_xyz, det_xyz, euler_abc_deg]
    marker_3d : (N_markers, 3) mm
    observed_2d : (N_markers, 2) [row, col] pixels, NaN for undetected
    weights : (N_markers,) unsharpness weights in (0, 1]
    anchor_idx : int — anchor marker (gets 100x weight)
    det_spacing : float mm
    det_rows, det_cols : int

    Returns
    -------
    residuals : (2 * N_valid,) float64 — weighted pixel residuals [row_err, col_err, ...]
    """
    cone_vec = params_to_cone_vec(params9, det_spacing)
    src     = cone_vec[0:3]
    det_ctr = cone_vec[3:6]
    u_vec   = cone_vec[6:9]
    v_vec   = cone_vec[9:12]

    pred = project_points_batch(marker_3d, src, det_ctr, u_vec, v_vec, det_rows, det_cols)

    # Anchor weight: 100x median of valid non-anchor weights
    valid_non_anchor_weights = [
        float(weights[j])
        for j in range(len(marker_3d))
        if j != anchor_idx and not np.any(np.isnan(observed_2d[j]))
    ]
    if valid_non_anchor_weights:
        w_anchor = 100.0 * float(np.median(valid_non_anchor_weights))
    else:
        w_anchor = 100.0 * float(np.max(weights))

    residuals = []
    for j in range(len(marker_3d)):
        obs = observed_2d[j]
        if np.any(np.isnan(obs)):
            continue

        p = pred[j]
        if np.any(np.isnan(p)):
            residuals.extend([1e3, 1e3])
            continue

        w = w_anchor if j == anchor_idx else float(weights[j])
        residuals.append(w * (p[0] - obs[0]))
        residuals.append(w * (p[1] - obs[1]))

    return np.array(residuals, dtype=np.float64)


def _unweighted_rms(params9, marker_3d, observed_2d, det_spacing, det_rows, det_cols):
    """Compute unweighted pixel RMS reprojection error (for reporting, not optimization)."""
    cone_vec = params_to_cone_vec(params9, det_spacing)
    pred = project_points_batch(
        marker_3d, cone_vec[0:3], cone_vec[3:6],
        cone_vec[6:9], cone_vec[9:12], det_rows, det_cols,
    )
    sq = []
    for j in range(len(marker_3d)):
        obs = observed_2d[j]
        if np.any(np.isnan(obs)) or np.any(np.isnan(pred[j])):
            continue
        sq.append((pred[j, 0] - obs[0])**2 + (pred[j, 1] - obs[1])**2)
    return float(np.sqrt(np.mean(sq))) if sq else np.inf


def solve_u7(
    marker_3d: np.ndarray,
    observed_2d: np.ndarray,
    weights: np.ndarray,
    anchor_idx: int,
    det_spacing: float,
    det_rows: int,
    det_cols: int,
    sod: float,
    odd: float,
    nominal_src: np.ndarray | None = None,
) -> dict:
    """
    Solve a single U7 problem via Levenberg-Marquardt (SDSG-005).

    Initial guess: per-shot nominal source position (if provided), else [0,0,-sod].
    Detector initial position derived as -src_nom * (odd/sod).

    Returns
    -------
    dict: params9, cost_px, success, nfev, message
    """
    if nominal_src is not None:
        src0 = np.asarray(nominal_src, dtype=np.float64)
        det0 = -src0 * (odd / sod)
    else:
        src0 = np.array([0.0, 0.0, -sod])
        det0 = np.array([0.0, 0.0,  odd])
    x0 = np.concatenate([src0, det0, [0.0, 0.0, 0.0]]).astype(np.float64)

    try:
        result = least_squares(
            fun=u7_residuals,
            x0=x0,
            args=(marker_3d, observed_2d, weights, anchor_idx,
                  det_spacing, det_rows, det_cols),
            method='lm',
            xtol=1e-10,
            ftol=1e-10,
            max_nfev=200,
            x_scale='jac',
        )

        params9  = result.x
        cost_px  = _unweighted_rms(params9, marker_3d, observed_2d, det_spacing, det_rows, det_cols)

        return {
            'params9': params9,
            'cost_px': cost_px,
            'success': bool(result.success),
            'nfev':    int(result.nfev),
            'message': result.message,
        }

    except Exception as e:
        return {
            'params9': x0.copy(),
            'cost_px': np.inf,
            'success': False,
            'nfev':    0,
            'message': str(e),
        }
