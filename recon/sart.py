"""
recon/sart.py — SIRT3D reconstruction via ASTRA SIRT3D_CUDA.

Uses ASTRA's built-in SIRT3D_CUDA algorithm (proper row+column normalisation)
rather than a manual per-shot update loop, which is equivalent for the
fully-simultaneous case and avoids the lam-sensitivity of a hand-rolled update.

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
    SIRT3D reconstruction (ASTRA SIRT3D_CUDA).

    Parameters
    ----------
    sinogram  : (det_rows, N_shots, det_cols) float32
    vectors   : (N_shots, 12) float64 (NaN rows are excluded)
    n_iter    : int — number of iterations
    lam       : float — relaxation factor passed to ASTRA (default 1.0)
    epsilon   : float — floor clamp applied after reconstruction
    voxel_size : float mm
    grid_size  : int — voxels per side (used when grid_nx/ny/nz are None)
    grid_nx, grid_ny, grid_nz : int | None — non-cubic grid overrides

    Returns
    -------
    volume      : (nx, ny, nz) float32 — (X,Y,Z)
    convergence : (n_iter,) float64 — zeros placeholder (ASTRA SIRT3D does not expose per-iter residuals)
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

    print(f"  SART: using {n_valid}/{n_shots} valid shots, grid {nx}x{ny}x{nz}, lam={lam}")

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

    vol_id  = astra.data3d.create('-vol',  vol_geom)
    proj_id = astra.data3d.create('-sino', proj_geom, data=sino_valid)

    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId']     = proj_id
    if lam != 1.0:
        cfg['option'] = {'relaxation_factor': float(lam)}

    alg_id = astra.algorithm.create(cfg)
    try:
        # Run in blocks of 10 to print progress
        block = 10
        for start in range(0, n_iter, block):
            count = min(block, n_iter - start)
            astra.algorithm.run(alg_id, count)
            print(f"    iter {start + count:3d}/{n_iter}")
        vol_zyx = astra.data3d.get(vol_id)   # (Z, Y, X)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    volume = vol_zyx.transpose(2, 1, 0).astype(np.float32)
    np.clip(volume, 0.0, None, out=volume)

    convergence = np.zeros(n_iter, dtype=np.float64)
    return volume, convergence
