"""
solver/projection.py — Coordinate utilities, pinhole projection, 9-DOF ↔ cone_vec conversion.

All conventions match forward/geometry.py (_rotation_matrix_zyx, _detector_basis) and
forward/centroiding.py (compute_ground_truth_projections) exactly so residuals vanish
at ground-truth parameters.
"""

import numpy as np


def _detector_basis(src_pos: np.ndarray):
    """
    Nominal (u, v) unit basis for a given source position.
    Identical to forward/geometry.py::_detector_basis.
    """
    beam = -src_pos / np.linalg.norm(src_pos)
    ref_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(beam, ref_up)) > 0.9:
        ref_up = np.array([0.0, 0.0, 1.0])
    u = np.cross(ref_up, beam)
    u /= np.linalg.norm(u)
    v = np.cross(beam, u)
    v /= np.linalg.norm(v)
    return u, v


def euler_to_R(euler_deg: np.ndarray) -> np.ndarray:
    """
    ZYX Euler rotation: R = Rz(c) @ Ry(b) @ Rx(a).
    Identical to forward/geometry.py::_rotation_matrix_zyx.
    """
    a, b, c = np.deg2rad(euler_deg)
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)

    Rx = np.array([[1,  0,   0],
                   [0,  ca, -sa],
                   [0,  sa,  ca]])
    Ry = np.array([[ cb, 0, sb],
                   [  0, 1,  0],
                   [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0],
                   [sc,  cc, 0],
                   [ 0,   0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R: np.ndarray) -> np.ndarray:
    """Extract ZYX Euler [a, b, c] in degrees from rotation matrix. Uses scipy for robustness."""
    from scipy.spatial.transform import Rotation as Rsc
    r = Rsc.from_matrix(R)
    zyx = r.as_euler('zyx', degrees=True)  # returns [c, b, a]
    return np.array([zyx[2], zyx[1], zyx[0]])


def params_to_cone_vec(params9: np.ndarray, det_spacing: float) -> np.ndarray:
    """
    Convert 9-DOF solver params to ASTRA cone_vec 12-element row.
    Mirrors forward/geometry.py::geometry_to_cone_vec exactly.

    params9: [src_x, src_y, src_z, det_x, det_y, det_z, euler_a, euler_b, euler_c]
    Returns: [srcX,srcY,srcZ, dX,dY,dZ, uX*h,uY*h,uZ*h, vX*h,vY*h,vZ*h]  (h = det_spacing)
    """
    src   = params9[0:3]
    det   = params9[3:6]
    euler = params9[6:9]

    u_nom, v_nom = _detector_basis(src)
    R     = euler_to_R(euler)
    u_rot = R @ u_nom
    v_rot = R @ v_nom

    cone_vec = np.empty(12, dtype=np.float64)
    cone_vec[0:3]  = src
    cone_vec[3:6]  = det
    cone_vec[6:9]  = u_rot * det_spacing
    cone_vec[9:12] = v_rot * det_spacing
    return cone_vec


def project_point(
    P: np.ndarray,
    src: np.ndarray,
    det_ctr: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    Pinhole-project 3D point P onto detector, returning (row, col) pixels.
    Matches forward/centroiding.py::compute_ground_truth_projections exactly.
    Returns (NaN, NaN) for rays parallel to the detector plane.
    """
    det_spacing = np.linalg.norm(u_vec)
    u_hat = u_vec / det_spacing
    v_hat = v_vec / det_spacing
    n_hat = np.cross(u_hat, v_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)

    direction = P - src
    denom = np.dot(n_hat, direction)

    if abs(denom) < 1e-10:
        return np.array([np.nan, np.nan])

    t = np.dot(n_hat, det_ctr - src) / denom
    P_det = src + t * direction

    delta = P_det - det_ctr
    u_mm  = np.dot(delta, u_hat)
    v_mm  = np.dot(delta, v_hat)

    col = u_mm / det_spacing + (det_cols - 1) / 2.0
    row = v_mm / det_spacing + (det_rows - 1) / 2.0

    return np.array([row, col])


def project_points_batch(
    points_3d: np.ndarray,
    src: np.ndarray,
    det_ctr: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    Vectorized pinhole projection for N points. Returns (N, 2) [row, col].
    NaN rows for degenerate rays.
    """
    det_spacing = np.linalg.norm(u_vec)
    u_hat = u_vec / det_spacing
    v_hat = v_vec / det_spacing
    n_hat = np.cross(u_hat, v_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)

    directions = points_3d - src[np.newaxis, :]       # (N, 3)
    denom = directions @ n_hat                         # (N,)

    nd_src = np.dot(n_hat, det_ctr - src)              # scalar
    degenerate = np.abs(denom) <= 1e-10
    safe_denom = np.where(degenerate, 1.0, denom)
    t = nd_src / safe_denom                            # (N,)

    P_det = src[np.newaxis, :] + t[:, np.newaxis] * directions  # (N, 3)
    delta = P_det - det_ctr[np.newaxis, :]

    u_mm = delta @ u_hat
    v_mm = delta @ v_hat

    col = u_mm / det_spacing + (det_cols - 1) / 2.0
    row = v_mm / det_spacing + (det_rows - 1) / 2.0

    out = np.column_stack([row, col])
    out[degenerate] = np.nan
    return out
