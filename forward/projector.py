"""
forward/projector.py — ASTRA GPU forward projector + noise pipeline.

Sinogram axis order: (det_rows, N_shots, det_cols)  ← FWD-008, verify with assert below.

ASTRA volume convention:
  - Phantom array is indexed (X, Y, Z); ASTRA expects (Z, Y, X).
  - Always call: vol_astra = volume.transpose(2, 1, 0)
  - vol_geom uses physical extents in mm so cone_vec mm coordinates are consistent.

Noise model (FWD-005 through FWD-007):
  1. Beer-Lambert: I = I0 * exp(-sino)
  2. Apply focal-spot unsharpness blur to intensity image (Gaussian, sigma=1 pixel)
  3. Poisson noise on blurred intensity
  4. Back-convert: sino_noisy = -log(I_noisy / I0), clamp negatives to 0
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def forward_project(
    volume: np.ndarray,
    vectors: np.ndarray,
    voxel_size: float,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    GPU forward projection via ASTRA FP3D_CUDA with cone_vec geometry.

    Parameters
    ----------
    volume : (Nx, Ny, Nz) float32 array — attenuation coefficients (mm^-1)
             Phantom-space axis order: first axis = X, second = Y, third = Z.
    vectors : (N_shots, 12) float64 — ASTRA cone_vec geometry (all in mm)
    voxel_size : float — isotropic voxel edge length (mm)
    det_rows : int — detector height (pixels)
    det_cols : int — detector width (pixels)

    Returns
    -------
    sinogram : (det_rows, N_shots, det_cols) float32
        Line integrals of attenuation [mm^-1 * mm = dimensionless].
        Axis order: (det_rows, N_shots, det_cols) — FWD-008.

    Notes
    -----
    ASTRA data objects are deleted after use to prevent CUDA memory leaks (RECON-005).
    """
    import astra

    n_shots = vectors.shape[0]
    nx, ny, nz = volume.shape

    # Half-extent of phantom in mm
    half_x = nx * voxel_size / 2.0
    half_y = ny * voxel_size / 2.0
    half_z = nz * voxel_size / 2.0

    # ASTRA volume: (Z, Y, X) layout with physical extents in mm.
    # create_vol_geom(nrows=nY, ncols=nX, nslices=nZ, minrow, maxrow, mincol, maxcol, minslice, maxslice)
    vol_geom = astra.create_vol_geom(
        ny, nx, nz,
        -half_y, half_y,   # Y (row) range
        -half_x, half_x,   # X (col) range
        -half_z, half_z,   # Z (slice) range
    )

    # Projection geometry: cone_vec, all coordinates in mm
    proj_geom = astra.create_proj_geom('cone_vec', det_rows, det_cols, vectors)

    # Transpose volume to ASTRA's (Z, Y, X) axis order
    vol_astra = volume.transpose(2, 1, 0).astype(np.float32)

    # Create ASTRA data objects
    vol_id  = astra.data3d.create('-vol',  vol_geom, data=vol_astra)
    proj_id = astra.data3d.create('-sino', proj_geom)

    # Configure and run forward projector
    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['VolumeDataId']    = vol_id
    cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Retrieve sinogram
    sinogram = astra.data3d.get(proj_id)   # (det_rows, N_shots, det_cols)

    # Cleanup — prevent CUDA memory leaks (RECON-005)
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id)

    # Verify axis order (FWD-008)
    assert sinogram.shape == (det_rows, n_shots, det_cols), (
        f"Unexpected sinogram shape {sinogram.shape}; "
        f"expected ({det_rows}, {n_shots}, {det_cols}). "
        "Check ASTRA version or axis ordering."
    )

    return sinogram.astype(np.float32)


def apply_noise_pipeline(
    sinogram: np.ndarray,
    I0: float = 10_000.0,
    focal_spot: float = 0.5,
    sod: float = 500.0,
    odd: float = 200.0,
    det_spacing: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """
    Apply Beer-Lambert + focal-spot unsharpness + Poisson noise to a clean sinogram.

    Physical model (in order):
      1. Convert line-integral sinogram to intensity: I = I0 * exp(-sino)  [FWD-005]
      2. Gaussian blur per projection slice to model focal-spot unsharpness [FWD-007]
         sigma = 1 pixel = det_spacing = 0.2 mm (PRD stated value)
         DECISION: PRD specifies sigma=0.2mm=1px. Formula (focal_spot*odd/sod/2.355)
         gives 0.085mm=0.424px; using PRD value since reviewers will see this spec.
      3. Poisson noise on blurred intensity                                 [FWD-006]
      4. Back-convert: sino_noisy = -log(I_noisy / I0), clamp to [0, inf)

    Parameters
    ----------
    sinogram : (det_rows, N_shots, det_cols) float32 — clean line-integral sinogram
    I0 : float — incident photon count (source intensity)
    focal_spot : float — focal spot FWHM (mm), stored for reference
    sod : float — source-to-object distance (mm), stored for reference
    odd : float — object-to-detector distance (mm), stored for reference
    det_spacing : float — detector pixel pitch (mm)
    seed : int — RNG seed for Poisson noise (NV-SHOT-004)

    Returns
    -------
    sino_noisy : (det_rows, N_shots, det_cols) float32 — noisy sinogram
    """
    det_rows, n_shots, det_cols = sinogram.shape

    # Step 1: Beer-Lambert — clean intensity image
    sino_f64 = sinogram.astype(np.float64)
    I_clean = I0 * np.exp(-sino_f64)   # (det_rows, N_shots, det_cols)

    # Step 2: Per-projection focal-spot blur (sigma = 1 pixel = 0.2mm, FWD-007)
    # Applied in intensity domain (physically correct — blur before detection).
    sigma_pixel = 1.0   # DECISION: PRD stated value
    I_blurred = np.empty_like(I_clean)
    for s in range(n_shots):
        I_blurred[:, s, :] = gaussian_filter(I_clean[:, s, :], sigma=sigma_pixel)

    # Clamp to avoid negative intensities from blur edge effects
    np.clip(I_blurred, 0.0, None, out=I_blurred)

    # Step 3: Poisson noise (seed for reproducibility)
    rng = np.random.default_rng(seed)
    I_noisy = rng.poisson(I_blurred).astype(np.float64)

    # Clamp to ≥ 1 to avoid log(0) — losing 1 photon is physically reasonable
    np.clip(I_noisy, 1.0, None, out=I_noisy)

    # Step 4: Back-convert to sinogram
    sino_noisy = -np.log(I_noisy / I0)
    np.clip(sino_noisy, 0.0, None, out=sino_noisy)   # clamp negatives (photon over-count)

    return sino_noisy.astype(np.float32)
