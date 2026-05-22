"""
recon/sart.py — SART (Simultaneous Algebraic Reconstruction Technique).

Algorithm (additive per-projection update):
  Initialize x = 0.05
  For each iteration:
    For each projection angle s:
      Ax_s   = forward(x, shot=s)
      delta_s = (b_s - Ax_s) / row_sum_s
      x      += lam * backproject(delta_s, shot=s) / col_sum_s
      x       = max(x, epsilon)

In practice we use a single-pass vectorised implementation:
  Ax     = full forward projection
  ratio  = b - Ax
  x     += lam * backproject(ratio) / col_sum   (standard SART approximation)

This matches SART convergence properties while keeping CUDA overhead low
(one FP + one BP per iteration, same as mART).

Volume axis order: (X,Y,Z) externally; (Z,Y,X) internally for ASTRA.
"""

import numpy as np


def reconstruct_sart(
    sinogram: np.ndarray,
    vectors: np.ndarray,
    n_iter: int = 50,
    lam: float = 1.0,
    epsilon: float = 1e-6,
    voxel_size: float = 0.1,
    grid_size: int = 250,
    grid_nx: int | None = None,
    grid_ny: int | None = None,
    grid_nz: int | None = None,
) -> tuple:
    """
    SART reconstruction.

    Parameters
    ----------
    sinogram  : (det_rows, N_shots, det_cols) float32
    vectors   : (N_shots, 12) float64 (NaN rows are excluded)
    n_iter    : int — number of iterations
    lam       : float — relaxation factor (default 1.0)
    epsilon   : float — floor clamp
    voxel_size : float mm
    grid_size  : int — voxels per side (used when grid_nx/ny/nz are None)
    grid_nx, grid_ny, grid_nz : int | None — non-cubic grid overrides

    Returns
    -------
    volume      : (nx, ny, nz) float32 — (X,Y,Z)
    convergence : (n_iter,) float64 — ||update|| / ||x|| per iteration
    """
    import astra

    det_rows, n_shots, det_cols = sinogram.shape

    nx = grid_nx if grid_nx is not None else grid_size
    ny = grid_ny if grid_ny is not None else grid_size
    nz = grid_nz if grid_nz is not None else grid_size

    # Filter NaN shots
    valid = ~np.any(np.isnan(vectors), axis=1)
    n_valid = int(valid.sum())
    sino_valid = sinogram[:, valid, :].astype(np.float32)
    vecs_valid = vectors[valid, :].astype(np.float64)

    print(f"  SART: using {n_valid}/{n_shots} valid shots, grid {nx}×{ny}×{nz}, lam={lam}")

    b = sino_valid.astype(np.float64)

    # ASTRA geometry
    half_x = nx * voxel_size / 2.0
    half_y = ny * voxel_size / 2.0
    half_z = nz * voxel_size / 2.0
    vol_geom  = astra.create_vol_geom(
        ny, nx, nz,
        -half_y, half_y,
        -half_x, half_x,
        -half_z, half_z,
    )
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vecs_valid)

    # Precompute col_sum = A^T * 1
    ones_sino = np.ones_like(sino_valid)
    col_sum = _backproject(ones_sino, vol_geom, proj_geom).astype(np.float64)
    col_sum = np.maximum(col_sum, epsilon)

    x = np.full((nx, ny, nz), 0.05, dtype=np.float64)
    convergence = np.zeros(n_iter, dtype=np.float64)

    for it in range(n_iter):
        Ax = _forward(x.astype(np.float32), vol_geom, proj_geom).astype(np.float64)

        residual = b - Ax
        update = _backproject(residual.astype(np.float32), vol_geom, proj_geom).astype(np.float64)
        update /= col_sum

        norm_x = np.linalg.norm(x)
        x = x + lam * update
        x = np.maximum(x, epsilon)

        convergence[it] = np.linalg.norm(lam * update) / (norm_x + epsilon)

        if (it + 1) % 10 == 0:
            print(f"    iter {it+1:3d}/{n_iter}  conv={convergence[it]:.6f}")

    return x.astype(np.float32), convergence


def _forward(vol_xyz: np.ndarray, vol_geom, proj_geom) -> np.ndarray:
    import astra
    vol_id  = astra.data3d.create('-vol',  vol_geom, data=vol_xyz.transpose(2, 1, 0).astype(np.float32))
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
