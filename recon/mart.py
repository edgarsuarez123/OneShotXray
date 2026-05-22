"""
recon/mart.py — mART (multiplicative ART) iterative reconstruction.

Algorithm (RECON-004):
  Initialize x = 0.05 (small positive value)
  Precompute col_sum = A^T * 1  (backproject all-ones sinogram)
  For each iteration:
    Ax   = forward(x)
    ratio = b / max(Ax, eps)
    x    = x * exp(backproject(log(ratio)) / col_sum)
    x    = max(x, eps)

All ASTRA objects created and destroyed per call to avoid CUDA leaks (RECON-005).
Volume axis order: (X, Y, Z) externally; (Z, Y, X) internally for ASTRA.
"""

import numpy as np


def _forward(vol_xyz: np.ndarray, vol_geom, proj_geom) -> np.ndarray:
    """
    ASTRA forward projection. Creates, runs, and deletes all ASTRA objects.

    Parameters
    ----------
    vol_xyz : (Nx, Ny, Nz) float32 — (X,Y,Z) convention

    Returns
    -------
    sino : (det_rows, N_shots, det_cols) float32
    """
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
    """
    ASTRA backprojection (BP3D_CUDA). Creates, runs, and deletes all ASTRA objects.

    Parameters
    ----------
    sino : (det_rows, N_shots, det_cols) float32

    Returns
    -------
    vol_xyz : (Nx, Ny, Nz) float32 — (X,Y,Z) convention
    """
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

    # (Z, Y, X) → (X, Y, Z)
    return vol_zyx.transpose(2, 1, 0).astype(np.float32)


def reconstruct_mart(
    sinogram: np.ndarray,
    vectors: np.ndarray,
    n_iter: int = 50,
    epsilon: float = 1e-6,
    voxel_size: float = 0.1,
    grid_size: int = 250,
    lam: float = 1.0,
    grid_nx: int | None = None,
    grid_ny: int | None = None,
    grid_nz: int | None = None,
) -> tuple:
    """
    mART reconstruction.

    Parameters
    ----------
    sinogram : (det_rows, N_shots, det_cols) float32
    vectors  : (N_shots, 12) float64 (may contain NaN rows)
    n_iter   : int — number of multiplicative iterations
    epsilon  : float — floor for clamping (prevents log(0))
    voxel_size : float mm
    grid_size  : int — voxels per side (used for all axes when grid_nx/ny/nz are None)
    lam      : float — relaxation factor applied to each multiplicative update (default 1.0)
    grid_nx, grid_ny, grid_nz : int | None — override individual axis sizes for non-cubic
               volumes (e.g. NIH 200×160×140). If None, grid_size is used for that axis.

    Returns
    -------
    volume       : (nx, ny, nz) float32 — (X,Y,Z) convention
    convergence  : (n_iter,) float64 — ||correction|| / ||x|| per iteration
    """
    import astra

    det_rows, n_shots, det_cols = sinogram.shape

    # Resolve grid dimensions (support non-cubic volumes)
    nx = grid_nx if grid_nx is not None else grid_size
    ny = grid_ny if grid_ny is not None else grid_size
    nz = grid_nz if grid_nz is not None else grid_size

    # Filter NaN shots
    valid = ~np.any(np.isnan(vectors), axis=1)
    n_valid = int(valid.sum())
    sino_valid = sinogram[:, valid, :].astype(np.float32)
    vecs_valid = vectors[valid, :].astype(np.float64)

    print(f"  mART: using {n_valid}/{n_shots} valid shots, grid {nx}×{ny}×{nz}, lam={lam}")

    # Clamp sinogram to epsilon (RECON-003)
    b = np.maximum(sino_valid.astype(np.float64), epsilon)

    # ASTRA geometry: create_vol_geom(nrows=nY, ncols=nX, nslices=nZ, ...)
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

    # Precompute col_sum = A^T * 1 (all-ones backprojection)
    ones_sino = np.ones_like(sino_valid)
    col_sum = _backproject(ones_sino, vol_geom, proj_geom).astype(np.float64)
    col_sum = np.maximum(col_sum, epsilon)

    # Initialize volume (must be > 0; 0.02 is close to brain/tissue attenuation at 70keV)
    x = np.full((nx, ny, nz), 0.05, dtype=np.float64)

    convergence = np.zeros(n_iter, dtype=np.float64)

    for it in range(n_iter):
        # Forward projection
        Ax = _forward(x.astype(np.float32), vol_geom, proj_geom).astype(np.float64)
        Ax = np.maximum(Ax, epsilon)

        # Multiplicative update in log domain
        log_ratio = np.log(b / Ax)
        correction = _backproject(log_ratio.astype(np.float32), vol_geom, proj_geom).astype(np.float64)
        correction /= col_sum

        x = x * np.exp(lam * correction)
        x = np.maximum(x, epsilon)

        # Convergence norm
        norm_x = np.linalg.norm(x)
        convergence[it] = np.linalg.norm(correction) / (norm_x + epsilon)

        if (it + 1) % 10 == 0:
            print(f"    iter {it+1:3d}/{n_iter}  conv={convergence[it]:.6f}")

    return x.astype(np.float32), convergence
