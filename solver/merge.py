"""
solver/merge.py — Fitness-weighted merge of U7 solutions with quaternion rotation averaging.

Uses Markley's eigenvector method (Markley et al. 2007) for quaternion averaging.
"""

import numpy as np
from scipy.spatial.transform import Rotation as Rsc


def euler_to_quat(euler_deg: np.ndarray) -> np.ndarray:
    """ZYX Euler [a, b, c] degrees → unit quaternion [w, x, y, z]."""
    r = Rsc.from_euler('zyx', [euler_deg[2], euler_deg[1], euler_deg[0]], degrees=True)
    q_xyzw = r.as_quat()   # scipy convention: [x, y, z, w]
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quat_to_euler(q_wxyz: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] → ZYX Euler [a, b, c] degrees."""
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
    r = Rsc.from_quat(q_xyzw)
    zyx = r.as_euler('zyx', degrees=True)  # [c, b, a]
    return np.array([zyx[2], zyx[1], zyx[0]])


def quaternion_average(quats: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Markley's eigenvector method for weighted quaternion averaging.

    Parameters
    ----------
    quats : (K, 4) float64 — [w, x, y, z] unit quaternions
    weights : (K,) float64 — non-negative weights, must sum > 0

    Returns
    -------
    q_avg : (4,) float64 — [w, x, y, z] unit quaternion
    """
    quats = quats.copy()
    # Hemisphere alignment: q and -q are the same rotation; align to q[0]
    q_ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], q_ref) < 0:
            quats[i] = -quats[i]

    # Build weighted outer-product accumulator
    M = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(quats, weights):
        M += w * np.outer(q, q)

    # Largest eigenvector is the optimal average quaternion
    _, eigvecs = np.linalg.eigh(M)   # ascending eigenvalues; eigh for symmetric
    q_avg = eigvecs[:, -1]
    return q_avg / np.linalg.norm(q_avg)


def merge_u7_solutions(
    results: list,
    failed_cost_threshold: float = 2.0,
) -> np.ndarray:
    """
    Fitness-weighted merge of N_markers U7 solutions into a single 9-DOF estimate.

    Softmin weights: w_k = exp(-J_k / J_min), normalized.
    Translations merged linearly; rotations merged via Markley quaternion average.

    Parameters
    ----------
    results : list of dicts from solve_u7(), each with keys:
              'params9', 'cost_px', 'success'
    failed_cost_threshold : float — U7s with cost_px > this are excluded (SDSG-009)

    Returns
    -------
    merged_params9 : (9,) float64
    """
    good = [
        r for r in results
        if r['params9'] is not None
        and r['success']
        and np.isfinite(r['cost_px'])
        and r['cost_px'] <= failed_cost_threshold
    ]

    if len(good) == 0:
        # All failed — return initial guess, flagged externally
        p0 = results[0]['params9']
        if p0 is not None and not np.any(np.isnan(p0)):
            sod = np.linalg.norm(p0[0:3])
            odd = np.linalg.norm(p0[3:6])
        else:
            sod, odd = 500.0, 200.0
        return np.array([0.0, 0.0, -sod, 0.0, 0.0, odd, 0.0, 0.0, 0.0])

    costs = np.array([r['cost_px'] for r in good])
    j_min = costs.min()

    # Softmin weights (adaptive temperature = J_min)
    raw_w = np.exp(-costs / max(j_min, 1e-10))
    weights = raw_w / raw_w.sum()

    params_arr = np.array([r['params9'] for r in good])   # (K, 9)

    # Merge source and detector translations
    merged_src = (weights[:, np.newaxis] * params_arr[:, 0:3]).sum(axis=0)
    merged_det = (weights[:, np.newaxis] * params_arr[:, 3:6]).sum(axis=0)

    # Merge rotations
    quats = np.array([euler_to_quat(params_arr[k, 6:9]) for k in range(len(good))])
    q_avg = quaternion_average(quats, weights)
    merged_euler = quat_to_euler(q_avg)

    return np.concatenate([merged_src, merged_det, merged_euler])
