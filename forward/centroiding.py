"""
forward/centroiding.py — Fiducial marker detection, sub-pixel refinement, accuracy measurement.

Detection pipeline for one projection slice (det_rows x det_cols):
  1. CENT-001: Background subtract + normalize sinogram so markers are bright blobs.
               Lead markers (mu=1.133) have ~2.04 extra line-integral above steel (mu=0.1149).
               In T_inv = exp(-sino) space, contrast is only ~5% — too weak for threshold=0.05.
               Instead: sino - gaussian_blur(sino, sigma=30) gives ~42% contrast. Much better.
  2. CENT-002: blob_log coarse detection (min_sigma=2, max_sigma=8, num_sigma=12, threshold=0.1)
  3. CENT-003: Nearest-neighbor identity matching to reprojected ground-truth positions
  4. CENT-004: 7-parameter Gaussian + linear background fit on raw sinogram (sub-pixel)
  5. CENT-005: RMS centroiding noise measurement (target < 0.15 px)
  6. CENT-006: Geometric unsharpness weights w = 1 / (1 + Ug / pixel_size)

Sinogram axis order: (det_rows, N_shots, det_cols) -- projection at shot i: sino[:, i, :]
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares
from skimage.feature import blob_log


# ---------------------------------------------------------------------------
# Single-projection functions
# ---------------------------------------------------------------------------

def detect_markers(
    projection: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 8.0,
    num_sigma: int = 12,
    threshold: float = 0.1,
) -> np.ndarray:
    """
    Detect fiducial marker blobs in a single sinogram projection slice (CENT-001/002).

    CENT-001 implementation: background-subtracted sinogram.
    Lead markers appear as bright bumps (~2 unit contrast) above the steel baseline.
    Subtracting a broad Gaussian background (sigma=30px) removes the slow-varying
    steel region, leaving the marker blobs clearly isolated for blob_log.

    Parameters
    ----------
    projection : (det_rows, det_cols) float array -- sinogram slice (line integrals)
    min_sigma, max_sigma, num_sigma : blob_log scale parameters (pixels)
    threshold : float -- blob_log absolute LoG response threshold

    Returns
    -------
    blobs : (K, 3) array -- [row, col, sigma] for each detected blob.
            Empty (0, 3) if nothing detected.
    """
    proj = projection.astype(np.float64)

    # CENT-001: subtract broad background to isolate marker-scale features.
    # sigma=30px removes the ~512px-scale phantom outline and steel gradient.
    background = gaussian_filter(proj, sigma=30.0)
    contrast = proj - background   # markers: positive bumps; uniform steel: ~0

    # Normalize to [0, 1]
    c_min, c_max = contrast.min(), contrast.max()
    if c_max <= c_min:
        return np.empty((0, 3), dtype=np.float64)
    contrast_norm = (contrast - c_min) / (c_max - c_min)

    blobs = blob_log(
        contrast_norm,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
    )  # returns (K, 3): [row, col, sigma]

    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return blobs.astype(np.float64)


def match_identities(
    detected_blobs: np.ndarray,
    expected_positions: np.ndarray,
    max_dist: float = 15.0,
) -> np.ndarray:
    """
    One-to-one nearest-neighbor match of blobs to expected marker positions (CENT-003).

    Greedy: sorts all (marker, blob) pairs by distance, assigns each blob to
    at most one marker and each marker to at most one blob. Markers with no blob
    within max_dist, or whose nearest blob was already claimed by a closer marker,
    are returned as NaN.

    This prevents a single blob (from two overlapping marker projections) from
    being incorrectly assigned to both markers, which would inflate centroiding
    noise and produce phantom detections with systematic ~2px errors.

    Parameters
    ----------
    detected_blobs : (K, 3) — [row, col, sigma] from blob_log
    expected_positions : (N_markers, 2) — [row, col] from ground-truth reprojection
    max_dist : float — maximum pixel distance for a valid match

    Returns
    -------
    matched : (N_markers, 2) float64 — matched blob [row, col] positions.
              NaN for unmatched markers (zero identity swap errors guaranteed: CENT-003).
    """
    n_markers = len(expected_positions)
    matched = np.full((n_markers, 2), np.nan, dtype=np.float64)

    if len(detected_blobs) == 0:
        return matched

    blob_rc = detected_blobs[:, :2]   # (K, 2): [row, col]
    n_blobs = len(blob_rc)

    # Build all (dist, marker_idx, blob_idx) pairs and sort by distance
    pairs = []
    for j, exp in enumerate(expected_positions):
        dists = np.linalg.norm(blob_rc - exp, axis=1)
        for k in range(n_blobs):
            if dists[k] <= max_dist:
                pairs.append((dists[k], j, k))
    pairs.sort()

    assigned_markers = set()
    assigned_blobs   = set()
    for dist, j, k in pairs:
        if j not in assigned_markers and k not in assigned_blobs:
            matched[j] = blob_rc[k]
            assigned_markers.add(j)
            assigned_blobs.add(k)

    return matched


def refine_centroids(
    projection: np.ndarray,
    coarse_positions: np.ndarray,
    window: int = 20,
    max_offset: float = 3.0,
) -> np.ndarray:
    """
    Sub-pixel refinement via 7-parameter Gaussian + linear background fit (CENT-004).

    Fits the model on the RAW sinogram (no background subtraction) to avoid
    cross-marker contamination from the sigma=30 background kernel.

    Model: f(r,c) = A * exp(-((r-r0)^2 + (c-c0)^2)/(2*sig^2))
                  + bg0 + bg_r*(r-rc) + bg_c*(c-cc)

    Parameters (7): [r0, c0, A, sig, bg0, bg_r, bg_c]
    - r0, c0  : sub-pixel centroid (bounded to seed +/- max_offset px)
    - A       : peak amplitude above background
    - sig     : Gaussian sigma (pixels); initial guess = 4.0 (1mm marker * 1.4x mag / 0.2mm pitch)
    - bg0     : background level at patch center
    - bg_r    : linear background gradient along row direction
    - bg_c    : linear background gradient along column direction

    Parameters
    ----------
    projection : (det_rows, det_cols) — raw sinogram slice (line integrals)
    coarse_positions : (N_markers, 2) — [row, col] seeds, may contain NaN
    window : int — half-width of the fit window in pixels
    max_offset : float — max allowed shift from seed (px); prevents jumping to neighbor

    Returns
    -------
    refined : (N_markers, 2) float64 — refined [row, col] positions.
              NaN preserved for undetected markers or failed fits.
    """
    det_rows, det_cols = projection.shape
    half = window // 2
    proj = projection.astype(np.float64)

    refined = np.full_like(coarse_positions, np.nan)

    for j, (r_seed, c_seed) in enumerate(coarse_positions):
        if np.isnan(r_seed) or np.isnan(c_seed):
            continue

        r0_int = int(round(r_seed))
        c0_int = int(round(c_seed))

        # Clip window to image bounds
        r_lo = max(0, r0_int - half)
        r_hi = min(det_rows, r0_int + half + 1)
        c_lo = max(0, c0_int - half)
        c_hi = min(det_cols, c0_int + half + 1)

        patch = proj[r_lo:r_hi, c_lo:c_hi]
        if patch.size == 0:
            continue

        # Grid coordinates relative to patch center (for linear gradient terms)
        pr = np.arange(r_lo, r_hi, dtype=np.float64)
        pc = np.arange(c_lo, c_hi, dtype=np.float64)
        RR, CC = np.meshgrid(pr, pc, indexing='ij')  # (n_r, n_c)

        rc = (r_lo + r_hi - 1) / 2.0  # patch center row
        cc = (c_lo + c_hi - 1) / 2.0  # patch center col

        patch_flat = patch.ravel()
        R_flat = RR.ravel()
        C_flat = CC.ravel()

        bg0_init = float(np.median(patch))
        A_init = float(patch.max()) - bg0_init
        if A_init <= 0:
            A_init = float(patch.max()) * 0.1 + 1e-6

        # x = [r0, c0, A, sig, bg0, bg_r, bg_c]
        x0 = np.array([r_seed, c_seed, A_init, 4.0, bg0_init, 0.0, 0.0])

        bounds_lo = [r_seed - max_offset, c_seed - max_offset, 0.01,  1.0, -np.inf, -np.inf, -np.inf]
        bounds_hi = [r_seed + max_offset, c_seed + max_offset, np.inf, 12.0,  np.inf,  np.inf,  np.inf]

        def residuals(x, R=R_flat, C=C_flat, data=patch_flat, rc=rc, cc=cc):
            r0, c0, A, sig, bg0, bg_r, bg_c = x
            model = (A * np.exp(-((R - r0)**2 + (C - c0)**2) / (2.0 * sig**2))
                     + bg0 + bg_r * (R - rc) + bg_c * (C - cc))
            return model - data

        try:
            result = least_squares(
                residuals,
                x0,
                bounds=(bounds_lo, bounds_hi),
                method='trf',
                max_nfev=300,
                ftol=1e-8,
                xtol=1e-8,
            )
            r_fit, c_fit, A_fit = result.x[0], result.x[1], result.x[2]

            # Reject if amplitude too small (no marker signal) — mark as undetected
            if A_fit < 0.01:
                refined[j] = np.nan
                continue

            refined[j, 0] = r_fit
            refined[j, 1] = c_fit

        except Exception:
            # Mark as undetected if optimizer fails
            refined[j] = np.nan

    return refined


# ---------------------------------------------------------------------------
# Geometry / ground-truth reprojection
# ---------------------------------------------------------------------------

def compute_ground_truth_projections(
    marker_3d: np.ndarray,
    vectors: np.ndarray,
    det_rows: int,
    det_cols: int,
) -> np.ndarray:
    """
    Pinhole-project 3D marker positions into 2D detector coordinates for each shot.

    Uses the same ASTRA cone_vec vectors as the forward projector so the
    coordinate systems are identical.

    Pixel center convention: pixel (0,0) is at detector corner; center of the full
    detector array is at ((det_rows-1)/2, (det_cols-1)/2) — consistent with ASTRA
    which places det_ctr at the physical center of pixel index (N-1)/2.

    Detector normal: n_hat = cross(u_hat, v_hat) — exact for any tilt angle.

    Parameters
    ----------
    marker_3d : (N_markers, 3) — marker positions in phantom coords (mm)
    vectors : (N_shots, 12) — ASTRA cone_vec [src, det, u_scaled, v_scaled]
    det_rows : int
    det_cols : int

    Returns
    -------
    gt_2d : (N_shots, N_markers, 2) float64 — [row, col] in pixel coordinates.
            Row 0 = top of detector, Col 0 = left.
    """
    n_shots    = len(vectors)
    n_markers  = len(marker_3d)
    gt_2d = np.zeros((n_shots, n_markers, 2), dtype=np.float64)

    for i in range(n_shots):
        src      = vectors[i, 0:3]
        det_ctr  = vectors[i, 3:6]
        u_vec    = vectors[i, 6:9]    # column direction * det_spacing (mm)
        v_vec    = vectors[i, 9:12]   # row direction * det_spacing (mm)

        det_spacing = np.linalg.norm(u_vec)   # pixel size in mm
        u_hat = u_vec / det_spacing
        v_hat = v_vec / det_spacing

        # Detector plane normal: exact for any tilt (not beam_unit approximation)
        n_hat = np.cross(u_hat, v_hat)
        n_hat = n_hat / np.linalg.norm(n_hat)

        # Plane equation: n_hat . (P - det_ctr) = 0
        # Ray: P = src + t * (m - src)
        # Solve for t: n_hat . (src + t*(m-src) - det_ctr) = 0

        for j, m in enumerate(marker_3d):
            direction = m - src
            denom = np.dot(n_hat, direction)

            if abs(denom) < 1e-10:
                # Ray nearly parallel to detector plane — place at center
                # DECISION: use (N-1)/2 pixel center convention (ASTRA-consistent)
                gt_2d[i, j] = [(det_rows - 1) / 2.0, (det_cols - 1) / 2.0]
                continue

            t = np.dot(n_hat, det_ctr - src) / denom
            P_det = src + t * direction   # intersection with detector plane

            # Detector-plane offset from center (mm)
            delta = P_det - det_ctr
            u_mm  = np.dot(delta, u_hat)   # mm along column direction
            v_mm  = np.dot(delta, v_hat)   # mm along row direction

            # ASTRA pixel center convention: center of detector = pixel (N-1)/2
            col = u_mm / det_spacing + (det_cols - 1) / 2.0
            row = v_mm / det_spacing + (det_rows - 1) / 2.0

            gt_2d[i, j] = [row, col]

    return gt_2d


# ---------------------------------------------------------------------------
# Co-projection detection
# ---------------------------------------------------------------------------

def _flag_coprojecting_markers(
    gt_2d_shot: np.ndarray,
    min_separation_px: float = 8.0,
) -> np.ndarray:
    """
    Flag markers that project too close to a neighbor for reliable centroiding.

    Marker projected diameter is ~14px (1mm radius * 1.4x mag / 0.2mm pitch),
    giving PSF sigma ~5px. Two PSFs within 14px cause >0.1px centroid bias on the
    7-parameter Gaussian fit (interference term exp(-d^2/(2*sigma^2)) * d > 0.15px
    for d < 14px at sigma=5px). Threshold 14px ensures bias < 0.1px.

    Parameters
    ----------
    gt_2d_shot : (N_markers, 2) — ground-truth [row, col] for one shot
    min_separation_px : float — minimum pixel separation required for reliability

    Returns
    -------
    valid : (N_markers,) bool — False for any marker in a pair closer than threshold.
            NaN positions are treated as invalid (False).
    """
    n_markers = len(gt_2d_shot)
    valid = np.ones(n_markers, dtype=bool)

    for j in range(n_markers):
        if np.any(np.isnan(gt_2d_shot[j])):
            valid[j] = False
            continue
        for k in range(j + 1, n_markers):
            if np.any(np.isnan(gt_2d_shot[k])):
                continue
            dist = np.linalg.norm(gt_2d_shot[j] - gt_2d_shot[k])
            if dist < min_separation_px:
                valid[j] = False
                valid[k] = False

    return valid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def measure_centroiding_noise(
    refined: np.ndarray,
    ground_truth_2d: np.ndarray,
) -> np.ndarray:
    """
    Compute per-shot RMS centroiding error in pixels (CENT-005).

    Parameters
    ----------
    refined : (N_shots, N_markers, 2) — refined centroid positions [row, col]
    ground_truth_2d : (N_shots, N_markers, 2) — ground-truth projections [row, col]

    Returns
    -------
    noise_per_shot : (N_shots,) float64 — RMS error (pixels) per shot.
                     NaN for shots with zero valid detections.
    """
    n_shots = refined.shape[0]
    noise_per_shot = np.full(n_shots, np.nan, dtype=np.float64)

    for i in range(n_shots):
        errors_sq = []
        for j in range(refined.shape[1]):
            r, c = refined[i, j]
            if np.isnan(r) or np.isnan(c):
                continue
            dr = r - ground_truth_2d[i, j, 0]
            dc = c - ground_truth_2d[i, j, 1]
            errors_sq.append(dr ** 2 + dc ** 2)
        if errors_sq:
            noise_per_shot[i] = np.sqrt(np.mean(errors_sq))

    return noise_per_shot


def compute_geometric_unsharpness(
    gt_9dof: np.ndarray,
    marker_3d: np.ndarray,
    focal_spot_mm: float,
) -> np.ndarray:
    """
    Per-shot, per-marker geometric unsharpness Ug = focal_spot * b / a (mm).

    a = along-beam distance from source to marker
    b = along-beam distance from marker to detector center

    Parameters
    ----------
    gt_9dof : (N_shots, 9) — [src_xyz, det_xyz, euler_abc]
    marker_3d : (N_markers, 3) — marker positions (mm)
    focal_spot_mm : float — focal spot diameter (mm)

    Returns
    -------
    Ug : (N_shots, N_markers) float64 — geometric unsharpness in mm
    """
    n_shots   = len(gt_9dof)
    n_markers = len(marker_3d)
    Ug = np.zeros((n_shots, n_markers), dtype=np.float64)

    for i in range(n_shots):
        src = gt_9dof[i, 0:3]
        det = gt_9dof[i, 3:6]
        beam = det - src
        beam_len = np.linalg.norm(beam)
        if beam_len < 1e-10:
            continue
        beam_unit = beam / beam_len

        for j, m in enumerate(marker_3d):
            a = np.dot(m - src, beam_unit)   # source → marker (mm along beam)
            b = np.dot(det - m, beam_unit)   # marker → detector (mm along beam)
            if a > 1e-6 and b > 1e-6:
                Ug[i, j] = focal_spot_mm * b / a
            else:
                Ug[i, j] = focal_spot_mm   # degenerate: marker at src or det

    return Ug


def compute_unsharpness_weights(Ug: np.ndarray, pixel_size: float) -> np.ndarray:
    """
    Convert geometric unsharpness to detection weights (CENT-006).

    w = 1 / (1 + Ug / pixel_size)

    Higher Ug → blurrier marker → lower weight in SDSG solver.

    Parameters
    ----------
    Ug : (N_shots, N_markers) — unsharpness in mm
    pixel_size : float — detector pixel pitch (mm)

    Returns
    -------
    weights : (N_shots, N_markers) float64 in (0, 1]
    """
    return 1.0 / (1.0 + Ug / pixel_size)


# ---------------------------------------------------------------------------
# High-level batch processor
# ---------------------------------------------------------------------------

def process_all_shots(
    sinogram: np.ndarray,
    gt_9dof: np.ndarray,
    vectors: np.ndarray,
    marker_3d: np.ndarray,
    det_spacing: float = 0.2,
    focal_spot_mm: float = 0.5,
    blob_kwargs: dict = None,
    refine_window: int = 20,
    min_separation_px: float = 14.0,
) -> dict:
    """
    Run the full centroiding pipeline for all shots.

    Parameters
    ----------
    sinogram : (det_rows, N_shots, det_cols) — noisy sinogram
    gt_9dof : (N_shots, 9) — ground-truth geometry
    vectors : (N_shots, 12) — ASTRA cone_vec (for reprojection)
    marker_3d : (N_markers, 3) — 3D marker positions (mm)
    det_spacing : float — pixel pitch (mm)
    focal_spot_mm : float — focal spot FWHM (mm)
    blob_kwargs : dict — overrides for detect_markers() defaults
    refine_window : int — half-window for Gaussian fit refinement
    min_separation_px : float — minimum separation to flag co-projecting markers

    Returns
    -------
    result : dict with keys:
        'positions'          : (N_shots, N_markers, 2) float64 — refined centroids [row, col]
        'ground_truth_2d'   : (N_shots, N_markers, 2) float64 — reprojected GT [row, col]
        'noise_per_shot'    : (N_shots,) float64 — per-shot RMS error (pixels)
        'unsharpness_weights': (N_shots, N_markers) float64
        'detection_mask'    : (N_shots, N_markers) bool — True where detected
        'n_detected'        : int — total detections (of N_shots * N_markers possible)
    """
    import tqdm as tqdm_mod

    det_rows, n_shots, det_cols = sinogram.shape
    n_markers = len(marker_3d)

    if blob_kwargs is None:
        blob_kwargs = {}

    # Ground-truth 2D projections for matching and seeding refinement
    gt_2d = compute_ground_truth_projections(marker_3d, vectors, det_rows, det_cols)

    # Output arrays
    positions     = np.full((n_shots, n_markers, 2), np.nan, dtype=np.float64)
    detection_mask = np.zeros((n_shots, n_markers), dtype=bool)

    for i in tqdm_mod.tqdm(range(n_shots), desc='Centroiding shots', unit='shot'):
        proj = sinogram[:, i, :]                          # (det_rows, det_cols)
        expected = gt_2d[i]                               # (N_markers, 2)

        # Flag co-projecting markers for this shot (exclude from refinement)
        separation_valid = _flag_coprojecting_markers(expected, min_separation_px)

        # Detect blobs (used only to confirm marker presence in coarse detection)
        blobs = detect_markers(proj, **blob_kwargs)       # (K, 3)

        # Match to expected positions
        coarse = match_identities(blobs, expected)        # (N_markers, 2) or NaN

        # Build seed array: use GT expected position as seed for stable Gaussian fit.
        # Blob position is only used to confirm detection (coarse not NaN).
        # This avoids blob_log's 1-2px positional uncertainty from seeding the fit.
        seed_positions = np.full_like(expected, np.nan)
        for j in range(n_markers):
            if not np.isnan(coarse[j, 0]) and separation_valid[j]:
                # DECISION: seed from GT expected position, not blob centroid.
                # Rationale: GT is exact; blob_log has ~1-2px positional uncertainty.
                # Gaussian fit bounds ±3px from seed prevent false convergence.
                seed_positions[j] = expected[j]

        # Refine using 7-param Gaussian fit seeded from GT expected positions
        refined = refine_centroids(proj, seed_positions, window=refine_window)

        positions[i] = refined
        detection_mask[i] = ~np.isnan(refined[:, 0])

    # Noise metrics
    noise_per_shot = measure_centroiding_noise(positions, gt_2d)

    # Unsharpness weights
    Ug      = compute_geometric_unsharpness(gt_9dof, marker_3d, focal_spot_mm)
    weights = compute_unsharpness_weights(Ug, det_spacing)

    n_detected = int(detection_mask.sum())

    return {
        'positions'           : positions,
        'ground_truth_2d'    : gt_2d,
        'noise_per_shot'     : noise_per_shot,
        'unsharpness_weights': weights,
        'detection_mask'     : detection_mask,
        'n_detected'         : n_detected,
    }
