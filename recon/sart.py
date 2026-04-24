"""
recon/sart.py — SART (Simultaneous Algebraic Reconstruction Technique).

Block-SART: process all projections together per iteration (SIRT-like update),
scaled by ray-sum normalizations. This is the additive ART variant required by
the NIH track.

Algorithm:
  Initialize x = 0.0
  Precompute W_col = A^T * 1   (backproject all-ones — column weights)
  Precompute W_row = A * 1     (forward project all-ones — row weights)
  For each iteration:
    Ax      = forward(x)
    diff    = (b - Ax) / max(W_row, eps)
    x = x + relaxation * backproject(diff) / max(W_col, eps)
    x = max(x, 0)   # non-negativity constraint

All ASTRA objects created and destroyed per call to avoid CUDA leaks (RECON-005).
Volume axis order: (X, Y, Z) externally; (Z, Y, X) internally for ASTRA.
"""

import numpy as np


def _forward(vol_xyz: np.ndarray, vol_geom, proj_geom) -> np.ndarray:
    import astra
    vol_astra = vol_xyz.transpose(2, 1, 0).astype(np.float32)
    vol_id  = astra.data3d.create('-vol',  vol_geom, data=vol_astra)
    proj_id = astra.data3d.create('-sino', proj_geom)
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId']     = vol_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        sino = astra.data3d.get(proj_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
    return sino.astype(np.float32)


def _backproject(sino: np.ndarray, vol_geom, proj_geom) -> np.ndarray:
    import astra
    proj_id = astra.data3d.create('-sino', proj_geom, data=sino.astype(np.float32))
    vol_id  = astra.data3d.create('-vol',  vol_geom)
    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId']     = proj_id
    alg_id = astra.algorithm.create(cfg)
    try:
        astra.algorithm.run(alg_id)
        vol_zyx = astra.data3d.get(vol_id)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)
    return vol_zyx.transpose(2, 1, 0).astype(np.float32)


def reconstruct_sart(
    sinogram: np.ndarray,
    vectors: np.ndarray,
    n_iter: int = 50,
    relaxation: float = 1.0,
    epsilon: float = 1e-6,
    voxel_size: float = 1.0,
    grid_size=250,
) -> tuple:
    """
    SART reconstruction (block-additive update).

    Parameters
    ----------
    sinogram   : (det_rows, N_shots, det_cols) float32
    vectors    : (N_shots, 12) float64 (may contain NaN rows)
    n_iter     : int
    relaxation : float — update scaling (0 < relaxation <= 2; 1.0 recommended)
    epsilon    : float — floor for denominators
    voxel_size : float mm
    grid_size  : int or (nx, ny, nz) tuple

    Returns
    -------
    volume      : (nx, ny, nz) float32 — (X,Y,Z)
    convergence : (n_iter,) float64 — ||correction||_2 / ||x||_2 per iteration
    """
    import astra

    if isinstance(grid_size, (tuple, list)):
        nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    else:
        nx = ny = nz = int(grid_size)

    det_rows, n_shots, det_cols = sinogram.shape

    # Filter NaN shots
    valid = ~np.any(np.isnan(vectors), axis=1)
    n_valid = int(valid.sum())
    sino_valid = sinogram[:, valid, :].astype(np.float32)
    vecs_valid = vectors[valid, :].astype(np.float64)

    print(f"  SART: using {n_valid}/{n_shots} valid shots")

    b = sino_valid.astype(np.float64)

    # ASTRA geometry
    half_x = nx * voxel_size / 2.0
    half_y = ny * voxel_size / 2.0
    half_z = nz * voxel_size / 2.0
    vol_geom = astra.create_vol_geom(
        ny, nx, nz,
        -half_y, half_y,
        -half_x, half_x,
        -half_z, half_z,
    )
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vecs_valid)

    # Column sums: W_col = A^T * 1  (all-ones sinogram backprojected)
    ones_sino = np.ones((det_rows, n_valid, det_cols), dtype=np.float32)
    W_col = _backproject(ones_sino, vol_geom, proj_geom).astype(np.float64)
    W_col = np.maximum(W_col, epsilon)

    # Row sums: W_row = A * 1  (forward project all-ones volume)
    ones_vol = np.ones((nx, ny, nz), dtype=np.float32)
    W_row = _forward(ones_vol, vol_geom, proj_geom).astype(np.float64)
    W_row = np.maximum(W_row, epsilon)

    # Initialize x = 0
    x = np.zeros((nx, ny, nz), dtype=np.float64)

    convergence = np.zeros(n_iter, dtype=np.float64)

    for k in range(n_iter):
        Ax   = _forward(x.astype(np.float32), vol_geom, proj_geom).astype(np.float64)
        diff = (b - Ax) / W_row
        bp   = _backproject(diff.astype(np.float32), vol_geom, proj_geom).astype(np.float64)

        correction = relaxation * bp / W_col
        x = x + correction
        x = np.maximum(x, 0.0)   # non-negativity

        norm_x = np.linalg.norm(x)
        convergence[k] = np.linalg.norm(correction) / max(norm_x, epsilon)

        if (k + 1) % 10 == 0 or k == n_iter - 1:
            print(f"    iter {k+1:3d}/{n_iter}  conv={convergence[k]:.6f}")

    astra.data3d.clear()
    return x.astype(np.float32), convergence
