"""
forward/nih_geometry.py — Shot geometry for NIH bedside CT (restricted arc).

ICU bedside constraint: the X-ray source can only access from one lateral side.
Modelled as a Fibonacci-sampled upper hemisphere with azimuth phi ∈ [0, π]
instead of [0, 2π], giving a 180° lateral arc of source positions.

Reuses perturb_geometry and geometry_to_cone_vec from forward/geometry.py.
"""

import numpy as np

_GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

# NIH scan parameters
SOD         = 500.0   # mm  source-to-object distance
ODD         = 200.0   # mm  object-to-detector distance
SIGMA_S     = 3.0     # mm  freehand translational uncertainty (larger than Navy)
SIGMA_THETA = 1.5     # deg freehand rotational uncertainty
N_SHOTS     = 80
DET_ROWS    = 512
DET_COLS    = 512
DET_SPACING = 0.4     # mm  (wider pitch than Navy — skull must fit in 204.8mm FOV)
I0          = 10_000  # photons/pixel at 70 keV


def generate_restricted_arc_shots(
    n_shots: int = N_SHOTS,
    sod: float = SOD,
) -> np.ndarray:
    """
    Fibonacci-sampled source positions on the upper hemisphere with azimuth
    restricted to phi ∈ [0, π] — ICU lateral access constraint.

    Y coordinate is always ≥ 0 (source stays on the +Y lateral side).

    Parameters
    ----------
    n_shots : int
    sod     : float — source-to-object distance (mm)

    Returns
    -------
    source_positions : (n_shots, 3) float64 — mm
    """
    i = np.arange(n_shots, dtype=np.float64)
    cos_theta = (i + 0.5) / n_shots          # uniform elevation, cos ∈ (0, 1)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    # Restrict phi to [0, π] — Fibonacci modulo π gives good coverage
    phi = (2.0 * np.pi * i / _GOLDEN_RATIO) % np.pi

    x = sod * sin_theta * np.cos(phi)
    y = sod * sin_theta * np.sin(phi)   # always ≥ 0 (half-arc)
    z = sod * cos_theta

    return np.column_stack([x, y, z])


def generate_full_arc_shots(
    n_shots: int = N_SHOTS,
    sod: float = SOD,
) -> np.ndarray:
    """
    Standard full-hemisphere Fibonacci sampling (comparison baseline).
    Identical to forward/geometry.py::generate_hemisphere_shots.
    """
    i = np.arange(n_shots, dtype=np.float64)
    cos_theta = (i + 0.5) / n_shots
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = 2.0 * np.pi * i / _GOLDEN_RATIO

    x = sod * sin_theta * np.cos(phi)
    y = sod * sin_theta * np.sin(phi)
    z = sod * cos_theta

    return np.column_stack([x, y, z])
