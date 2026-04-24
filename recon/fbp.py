"""
recon/fbp.py — FBP reconstruction via ASTRA FDK_CUDA.

Filters NaN shots from recovered_cone_vec before reconstruction.
Volume axis order: (X, Y, Z) — same as phantom convention.
ASTRA internal order: (Z, Y, X) — transpose applied on input/output.
"""

import numpy as np


def reconstruct_fbp(
    sinogram: np.ndarray,
    vectors: np.ndarray,
    voxel_size: float = 0.1,
    grid_size=250,
) -> np.ndarray:
    """
    FBP (FDK_CUDA) reconstruction from cone_vec sinogram.

    Parameters
    ----------
    sinogram : (det_rows, N_shots, det_cols) float32 — noisy sinogram
    vectors  : (N_shots, 12) float64 — ASTRA cone_vec (may contain NaN rows)
    voxel_size : float mm — isotropic voxel pitch
    grid_size  : int or (nx, ny, nz) tuple — voxels per side (cubic or non-cubic)

    Returns
    -------
    volume : (nx, ny, nz) float32 — attenuation (mm^-1), (X,Y,Z)
    """
    import astra

    # Support non-cubic volumes
    if isinstance(grid_size, (tuple, list)):
        nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])
    else:
        nx = ny = nz = int(grid_size)

    # ASTRA FDK_CUDA requires nx == ny == nz (cubic grid).
    # For non-cubic targets, reconstruct on a cubic grid that encloses the target volume,
    # then crop the result back to (nx, ny, nz).
    # DECISION: use n_cubic = max(nx, ny, nz) as the cubic side length.
    n_cubic = max(nx, ny, nz)
    is_noncubic = (nx != ny or ny != nz)
    nx_recon = ny_recon = nz_recon = n_cubic

    det_rows, n_shots, det_cols = sinogram.shape

    # Filter NaN shots (solver failed → recovered_cone_vec row is NaN)
    valid = ~np.any(np.isnan(vectors), axis=1)
    n_valid = int(valid.sum())
    sino_valid  = sinogram[:, valid, :].astype(np.float32)
    vecs_valid  = vectors[valid, :].astype(np.float64)

    print(f"  FBP: using {n_valid}/{n_shots} valid shots (filtered {n_shots - n_valid} NaN rows)")
    if is_noncubic:
        print(f"  FBP: non-cubic grid {nx}x{ny}x{nz} -> cubic {n_cubic}^3 for FDK, then crop")

    # Volume geometry — extents in mm (always cubic for FDK)
    half = n_cubic * voxel_size / 2.0
    vol_geom = astra.create_vol_geom(
        ny_recon, nx_recon, nz_recon,
        -half, half,   # Y range
        -half, half,   # X range
        -half, half,   # Z range
    )

    # Projection geometry
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vecs_valid)

    # ASTRA expects (Z, Y, X)
    vol_id  = astra.data3d.create('-vol', vol_geom)
    proj_id = astra.data3d.create('-sino', proj_geom, data=sino_valid)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = vol_id
    cfg['ProjectionDataId']     = proj_id
    alg_id = astra.algorithm.create(cfg)

    try:
        astra.algorithm.run(alg_id)
        vol_zyx = astra.data3d.get(vol_id)   # (Z, Y, X) of shape (nz_recon, ny_recon, nx_recon)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    # Transpose (Z, Y, X) -> (X, Y, Z)
    volume = vol_zyx.transpose(2, 1, 0).astype(np.float32)
    np.clip(volume, 0.0, None, out=volume)   # FBP can produce small negatives

    # Crop from cubic to target non-cubic dimensions (centered crop)
    if is_noncubic:
        cx = (n_cubic - nx) // 2
        cy = (n_cubic - ny) // 2
        cz = (n_cubic - nz) // 2
        volume = volume[cx:cx + nx, cy:cy + ny, cz:cz + nz]

    return volume  # (nx, ny, nz)
