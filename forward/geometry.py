"""
forward/geometry.py — Shot position generation and ASTRA cone_vec conversion.

Coordinate convention (all in mm):
  - Origin = phantom center
  - X = right, Y = up (within phantom), Z = toward source (positive hemisphere)
  - Source positions on upper (+Z) hemisphere at radius SOD
  - Detector center on opposite side at radius ODD

ASTRA cone_vec 12-element format per shot:
  [srcX, srcY, srcZ,   -- source position
   dX,   dY,   dZ,     -- detector center position
   uX,   uY,   uZ,     -- detector column direction vector (scaled by pixel size)
   vX,   vY,   vZ]     -- detector row direction vector (scaled by pixel size)

Sinogram axis order: (det_rows, N_shots, det_cols)
"""

import numpy as np

_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hemisphere_shots(n_shots: int, sod: float, seed: int = 42) -> np.ndarray:
    """
    Fibonacci hemisphere sampling of source positions on the +Z hemisphere.

    Samples cos(theta) uniformly in [0, 1] with Fibonacci azimuthal spacing,
    giving a near-uniform distribution over the upper hemisphere (NV-SHOT-002).

    Parameters
    ----------
    n_shots : int
        Number of source positions to generate.
    sod : float
        Source-to-object distance (mm). All positions are at radius = sod.
    seed : int
        Unused (Fibonacci is deterministic), kept for API consistency.

    Returns
    -------
    source_positions : (n_shots, 3) float64 array, in mm.
    """
    i = np.arange(n_shots, dtype=np.float64)
    # Uniform in cos(theta) over upper hemisphere → z > 0 guaranteed
    cos_theta = (i + 0.5) / n_shots       # in (0, 1)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = 2.0 * np.pi * i / _GOLDEN_RATIO  # full 360° azimuth (NV-SHOT-002)

    x = sod * sin_theta * np.cos(phi)
    y = sod * sin_theta * np.sin(phi)
    z = sod * cos_theta

    return np.column_stack([x, y, z])      # (N, 3)


def perturb_geometry(
    source_pos: np.ndarray,
    sod: float,
    odd: float,
    sigma_s: float,
    sigma_theta: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Add per-shot freehand positioning uncertainty to nominal geometry (FWD-004).

    For each shot:
      - Source XYZ perturbed by N(0, sigma_s) mm in each axis
      - Detector center perturbed by N(0, sigma_s) mm in each axis
      - Detector orientation perturbed by N(0, sigma_theta) degrees (ZYX Euler)

    Parameters
    ----------
    source_pos : (N, 3) array of nominal source positions (mm)
    sod : float — source-to-object distance (mm), used to compute nominal det center
    odd : float — object-to-detector distance (mm)
    sigma_s : float — position noise std dev (mm)
    sigma_theta : float — orientation noise std dev (degrees)
    seed : int — RNG seed for reproducibility (NV-SHOT-004)

    Returns
    -------
    gt_9dof : (N, 9) float64 array
        Columns: [src_x, src_y, src_z, det_x, det_y, det_z, euler_a, euler_b, euler_c]
        Euler angles in degrees (ZYX convention: a=roll, b=pitch, c=yaw).
    """
    rng = np.random.default_rng(seed)
    n = len(source_pos)

    # Perturb source positions
    src_perturbed = source_pos + rng.normal(0.0, sigma_s, (n, 3))

    # Nominal detector centers: opposite side from source at ODD from origin
    src_unit = source_pos / np.linalg.norm(source_pos, axis=1, keepdims=True)
    det_nominal = -odd * src_unit   # (N, 3)

    # Perturb detector centers with same sigma_s
    det_perturbed = det_nominal + rng.normal(0.0, sigma_s, (n, 3))

    # Per-shot Euler angle perturbations (degrees)
    euler_perturbed = rng.normal(0.0, sigma_theta, (n, 3))

    gt_9dof = np.column_stack([src_perturbed, det_perturbed, euler_perturbed])
    return gt_9dof.astype(np.float64)    # (N, 9)


def geometry_to_cone_vec(
    gt_9dof: np.ndarray,
    det_spacing: float,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    Convert ground-truth 9-DOF geometry to ASTRA cone_vec 12-element format (FWD-001).

    For each shot, the nominal detector basis (u, v) is derived from the source
    direction, then rotated by the shot's Euler perturbation angles.

    Parameters
    ----------
    gt_9dof : (N, 9) array — [src_xyz, det_xyz, euler_abc] (mm, mm, degrees)
    det_spacing : float — detector pixel pitch (mm)
    det_rows : int — number of detector rows
    det_cols : int — number of detector columns

    Returns
    -------
    vectors : (N, 12) float64 array — ASTRA cone_vec format
        [srcX,srcY,srcZ, dX,dY,dZ, uX,uY,uZ, vX,vY,vZ]
        u = column direction * det_spacing (mm)
        v = row direction * det_spacing (mm)
    """
    n = len(gt_9dof)
    vectors = np.zeros((n, 12), dtype=np.float64)

    for i in range(n):
        src    = gt_9dof[i, 0:3]
        det    = gt_9dof[i, 3:6]
        euler  = gt_9dof[i, 6:9]

        u_nom, v_nom = _detector_basis(src)

        # Apply Euler rotation (perturbation of detector orientation)
        R = _rotation_matrix_zyx(euler)
        u_rot = R @ u_nom
        v_rot = R @ v_nom

        vectors[i, 0:3]  = src
        vectors[i, 3:6]  = det
        vectors[i, 6:9]  = u_rot * det_spacing   # column direction, scaled (FWD-001)
        vectors[i, 9:12] = v_rot * det_spacing   # row direction, scaled

    return vectors


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detector_basis(src_pos: np.ndarray):
    """
    Compute nominal detector (u, v) unit basis vectors for a given source position.

    u = detector column direction (horizontal)
    v = detector row direction (vertical)
    Both perpendicular to the beam axis (src → origin).

    Uses a stable reference-up approach to avoid gimbal singularity.
    """
    beam = -src_pos / np.linalg.norm(src_pos)   # unit vector from src toward origin

    # Choose a reference "up" not parallel to beam
    ref_up = np.array([0.0, 1.0, 0.0])          # Y-up default
    if abs(np.dot(beam, ref_up)) > 0.9:
        ref_up = np.array([0.0, 0.0, 1.0])      # switch to Z-up if beam near Y

    # u = horizontal column direction (right when looking in beam direction)
    u = np.cross(ref_up, beam)
    u /= np.linalg.norm(u)

    # v = vertical row direction, completing right-hand system in detector plane
    v = np.cross(beam, u)
    v /= np.linalg.norm(v)

    return u, v


def _rotation_matrix_zyx(euler_deg: np.ndarray) -> np.ndarray:
    """
    ZYX Euler rotation matrix from (roll=a, pitch=b, yaw=c) in degrees.
    R = Rz(c) @ Ry(b) @ Rx(a)
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
