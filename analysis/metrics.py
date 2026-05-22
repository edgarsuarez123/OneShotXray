"""
analysis/metrics.py — Image quality metrics for SDSG reconstructions.

CRITICAL: Always pass data_range explicitly to SSIM and PSNR.
          Default silently halves SSIM for float arrays (CLAUDE.md note).

Crack geometry (Navy phantom):
  - All cracks at Y = -0.5mm, voxel iy = 120 (grid_size=250, voxel_size=0.1mm)
  - crack_center_y_mm = -12.5 + 12.0 = -0.5mm
  - Widths: 0.4mm, 0.8mm, 1.6mm (wider overwrites narrower in phantom)
"""

import numpy as np


# ── Crack geometry constants (from navy_phantom.py) ───────────────────────────
_CRACK_CENTER_Y_MM  = -0.5      # -12.5 + 12.0mm depth
_CRACK_WIDTHS_MM    = [0.4, 0.8, 1.6]
_BG_OFFSET_MM       = 5.0       # BG slab offset from crack center


def compute_ssim(recon: np.ndarray, ref: np.ndarray) -> float:
    """
    Structural similarity index (SSIM).

    IMPORTANT: data_range = ref.max() - ref.min() must be passed explicitly.
    Omitting it silently halves SSIM for normalized float arrays.

    Parameters
    ----------
    recon : (Nx, Ny, Nz) float32 — reconstructed volume
    ref   : (Nx, Ny, Nz) float32 — reference (inpainted phantom)

    Returns
    -------
    ssim_val : float in [0, 1]
    """
    from skimage.metrics import structural_similarity

    data_range = float(ref.max() - ref.min())
    if data_range < 1e-12:
        return 1.0

    ssim_val = structural_similarity(
        recon.astype(np.float32),
        ref.astype(np.float32),
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    return float(ssim_val)


def compute_psnr(recon: np.ndarray, ref: np.ndarray) -> float:
    """
    Peak signal-to-noise ratio (PSNR) in dB.

    IMPORTANT: data_range must be passed explicitly.

    Parameters
    ----------
    recon : (Nx, Ny, Nz) float32
    ref   : (Nx, Ny, Nz) float32

    Returns
    -------
    psnr_val : float dB
    """
    from skimage.metrics import peak_signal_noise_ratio

    data_range = float(ref.max() - ref.min())
    if data_range < 1e-12:
        return np.inf

    psnr_val = peak_signal_noise_ratio(
        ref.astype(np.float32),
        recon.astype(np.float32),
        data_range=data_range,
    )
    return float(psnr_val)


def compute_cnr(
    recon: np.ndarray,
    defect_mask: np.ndarray,
    bg_mask: np.ndarray,
) -> float:
    """
    Contrast-to-noise ratio: |mean(defect) - mean(bg)| / std(bg).

    Parameters
    ----------
    recon       : (Nx, Ny, Nz) float32 — reconstruction or phantom
    defect_mask : (Nx, Ny, Nz) bool — voxels in defect (crack) region
    bg_mask     : (Nx, Ny, Nz) bool — voxels in homogeneous background

    Returns
    -------
    cnr : float
    """
    defect_vals = recon[defect_mask].astype(np.float64)
    bg_vals     = recon[bg_mask].astype(np.float64)

    if len(defect_vals) == 0 or len(bg_vals) == 0:
        return 0.0

    bg_std = float(np.std(bg_vals))
    if bg_std < 1e-12:
        return 0.0

    return float(abs(np.mean(defect_vals) - np.mean(bg_vals)) / bg_std)


def build_crack_masks(
    grid_size: int = 250,
    voxel_size: float = 0.1,
    dilation: int = 3,
) -> dict:
    """
    Build defect and background masks for each crack width.

    The crack is a Y-aligned slab at Y=-0.5mm (iy=120).
    Defect mask: Y-slab of crack_width, restricted to central XZ region.
    BG mask: same-size Y-slab offset 5mm in +Y, same XZ region.

    Parameters
    ----------
    grid_size  : int — voxels per side (250)
    voxel_size : float mm (0.1)
    dilation   : int voxels — extra margin around crack for defect mask

    Returns
    -------
    masks : dict mapping crack_width_mm (float) → (defect_mask, bg_mask)
            each mask is (grid_size, grid_size, grid_size) bool
    """
    n  = grid_size
    vs = voxel_size

    # Crack center in voxel space
    crack_iy = int(round(_CRACK_CENTER_Y_MM / vs + n / 2.0))

    # Central XZ region: avoid phantom edges and far-corner marker artifacts
    # Markers at ±9mm → ±90 vox from center (idx 35–215)
    # Use central 60% of volume: [50, 200) in X and Z
    margin = int(n * 0.20)   # 50 voxels on each side
    x0, x1 = margin, n - margin
    z0, z1 = margin, n - margin

    bg_offset_vox = int(round(_BG_OFFSET_MM / vs))

    masks = {}
    for width_mm in _CRACK_WIDTHS_MM:
        half_vox = max(1, int(np.ceil((width_mm / 2.0) / vs)))
        hw = half_vox + dilation

        iy_lo = max(0, crack_iy - hw)
        iy_hi = min(n - 1, crack_iy + hw)

        defect = np.zeros((n, n, n), dtype=bool)
        defect[x0:x1, iy_lo:iy_hi + 1, z0:z1] = True

        bg_iy_lo = iy_lo + bg_offset_vox
        bg_iy_hi = iy_hi + bg_offset_vox
        bg_iy_hi = min(bg_iy_hi, n - 1)

        bg = np.zeros((n, n, n), dtype=bool)
        if bg_iy_lo < n:
            bg[x0:x1, bg_iy_lo:bg_iy_hi + 1, z0:z1] = True

        masks[width_mm] = (defect, bg)

    return masks
