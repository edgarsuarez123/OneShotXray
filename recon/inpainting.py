"""
recon/inpainting.py — Fiducial marker removal via annular shell-mean inpainting.

For each marker:
  1. Build sphere mask (r <= marker_radius_mm)
  2. Build annular shell mask (marker_radius_mm < r <= shell_outer_mm)
  3. Fill sphere voxels with mean of shell voxels (local background estimate)

All operations on a copy — original volume is not modified.
"""

import numpy as np


def inpaint_markers(
    volume: np.ndarray,
    marker_positions_mm: np.ndarray,
    marker_radius_mm: float = 1.5,
    shell_outer_mm: float = 4.5,
    voxel_size: float = 0.1,
    grid_size: int = 250,
) -> np.ndarray:
    """
    Remove fiducial marker artifacts via shell-mean inpainting.

    Parameters
    ----------
    volume : (grid_size, grid_size, grid_size) float32 — (X,Y,Z)
    marker_positions_mm : (N_markers, 3) float32/64 — marker centers in mm
    marker_radius_mm : float — inpainting sphere radius (should be >= actual marker radius)
    shell_outer_mm   : float — outer radius of background-estimation shell
    voxel_size : float mm
    grid_size  : int — voxels per side (isotropic)

    Returns
    -------
    inpainted : (grid_size, grid_size, grid_size) float32 — copy with markers removed
    """
    result = volume.copy()
    n = grid_size
    vs = voxel_size

    for pos_mm in marker_positions_mm:
        cx, cy, cz = float(pos_mm[0]), float(pos_mm[1]), float(pos_mm[2])

        # Convert center to voxel index (same formula as phantom builder)
        icx = cx / vs + n / 2.0
        icy = cy / vs + n / 2.0
        icz = cz / vs + n / 2.0

        # Bounding box covering shell_outer_mm
        pad = int(np.ceil(shell_outer_mm / vs)) + 1
        ix0 = max(0, int(icx) - pad)
        ix1 = min(n, int(icx) + pad + 1)
        iy0 = max(0, int(icy) - pad)
        iy1 = min(n, int(icy) + pad + 1)
        iz0 = max(0, int(icz) - pad)
        iz1 = min(n, int(icz) + pad + 1)

        # Voxel centers relative to marker center (mm)
        xs = (np.arange(ix0, ix1) + 0.5 - n / 2.0) * vs - cx
        ys = (np.arange(iy0, iy1) + 0.5 - n / 2.0) * vs - cy
        zs = (np.arange(iz0, iz1) + 0.5 - n / 2.0) * vs - cz

        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
        r2 = XX**2 + YY**2 + ZZ**2

        sphere_mask = r2 <= marker_radius_mm**2
        shell_mask  = (r2 > marker_radius_mm**2) & (r2 <= shell_outer_mm**2)

        # Extract the sub-volume patch
        patch = result[ix0:ix1, iy0:iy1, iz0:iz1]

        # Background estimate from annular shell
        shell_vals = patch[shell_mask]
        bg_mean = float(shell_vals.mean()) if len(shell_vals) > 0 else 0.0

        # Fill sphere with local background
        patch[sphere_mask] = bg_mean

    return result
