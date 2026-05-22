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
    grid_size: int = 250,
) -> np.ndarray:
    """
    FBP (FDK_CUDA) reconstruction from cone_vec sinogram.

    Parameters
    ----------
    sinogram : (det_rows, N_shots, det_cols) float32 — noisy sinogram
    vectors  : (N_shots, 12) float64 — ASTRA cone_vec (may contain NaN rows)
    voxel_size : float mm — isotropic voxel pitch
    grid_size  : int — voxels per side

    Returns
    -------
    volume : (grid_size, grid_size, grid_size) float32 — attenuation (mm^-1), (X,Y,Z)
    """
    import astra

    det_rows, n_shots, det_cols = sinogram.shape

    # Filter NaN shots (solver failed → recovered_cone_vec row is NaN)
    valid = ~np.any(np.isnan(vectors), axis=1)
    n_valid = int(valid.sum())
    sino_valid  = sinogram[:, valid, :].astype(np.float32)
    vecs_valid  = vectors[valid, :].astype(np.float64)

    print(f"  FBP: using {n_valid}/{n_shots} valid shots (filtered {n_shots - n_valid} NaN rows)")

    # Volume geometry — match forward/projector.py:63-68 exactly
    half = grid_size * voxel_size / 2.0
    vol_geom = astra.create_vol_geom(
        grid_size, grid_size, grid_size,
        -half, half,   # Y (row) range
        -half, half,   # X (col) range
        -half, half,   # Z (slice) range
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
        vol_zyx = astra.data3d.get(vol_id)   # (Z, Y, X)
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(proj_id)

    # Transpose (Z, Y, X) → (X, Y, Z) to match phantom convention
    volume = vol_zyx.transpose(2, 1, 0).astype(np.float32)
    np.clip(volume, 0.0, None, out=volume)   # FBP can produce small negatives

    return volume
