"""
diag3_formula.py - Test corrected GT projection formula.

H3: Using actual detector plane normal n = cross(u_hat, v_hat) instead of
    beam_unit. For perturbed geometry (Euler rotation), these differ.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

HDF5    = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
PHANTOM = ROOT / 'data' / 'navy' / 'phantom.h5'

with h5py.File(HDF5, 'r') as f:
    sinogram = f['sinogram'][:]
    vectors  = f['shots/nominal/cone_vec'][:]
    blob_pos = f['centroids/positions'][:]

with h5py.File(PHANTOM, 'r') as f:
    marker_3d  = f['marker_positions'][:].astype(np.float64)

DET_ROWS, N_SHOTS, DET_COLS = sinogram.shape
N_MARKERS = len(marker_3d)


def project_v3(marker_3d, vectors, det_rows, det_cols):
    """
    Corrected formula: uses actual detector plane normal n = cross(u_hat, v_hat)
    instead of approximate beam_unit. Also uses (N-1)/2 pixel center offset.
    """
    n_shots = len(vectors)
    n_m     = len(marker_3d)
    gt_2d   = np.zeros((n_shots, n_m, 2), dtype=np.float64)
    col_ctr = (det_cols - 1) / 2.0
    row_ctr = (det_rows - 1) / 2.0

    for i in range(n_shots):
        src     = vectors[i, 0:3]
        det_ctr = vectors[i, 3:6]
        u_vec   = vectors[i, 6:9]
        v_vec   = vectors[i, 9:12]

        det_spacing = np.linalg.norm(u_vec)
        u_hat = u_vec / det_spacing
        v_hat = v_vec / det_spacing

        # Correct detector plane normal (not beam direction)
        n_hat = np.cross(u_hat, v_hat)
        n_hat /= np.linalg.norm(n_hat)

        # Signed distance from src to detector plane along n_hat
        t_det_plane = np.dot(n_hat, det_ctr - src)

        for j, m in enumerate(marker_3d):
            direction = m - src
            denom = np.dot(n_hat, direction)
            if abs(denom) < 1e-10:
                gt_2d[i, j] = [row_ctr, col_ctr]
                continue
            t = t_det_plane / denom
            P_det = src + t * direction
            delta = P_det - det_ctr
            col = np.dot(delta, u_hat) / det_spacing + col_ctr
            row = np.dot(delta, v_hat) / det_spacing + row_ctr
            gt_2d[i, j] = [row, col]

    return gt_2d


def project_v2(marker_3d, vectors, det_rows, det_cols):
    """V2: beam_unit as plane normal, (N-1)/2 center."""
    n_shots = len(vectors); n_m = len(marker_3d)
    gt_2d = np.zeros((n_shots, n_m, 2), dtype=np.float64)
    col_ctr = (det_cols - 1) / 2.0; row_ctr = (det_rows - 1) / 2.0
    for i in range(n_shots):
        src = vectors[i, 0:3]; det_ctr = vectors[i, 3:6]
        u_vec = vectors[i, 6:9]; v_vec = vectors[i, 9:12]
        ds = np.linalg.norm(u_vec); u_hat = u_vec/ds; v_hat = v_vec/ds
        beam_unit = (det_ctr - src) / np.linalg.norm(det_ctr - src)
        for j, m in enumerate(marker_3d):
            d = m - src; denom = np.dot(beam_unit, d)
            if abs(denom) < 1e-10:
                gt_2d[i, j] = [row_ctr, col_ctr]; continue
            t = np.dot(beam_unit, det_ctr - src) / denom
            P = src + t * d; delta = P - det_ctr
            gt_2d[i, j] = [np.dot(delta, v_hat)/ds + row_ctr, np.dot(delta, u_hat)/ds + col_ctr]
    return gt_2d


def rms_errors(positions, gt_2d):
    errors = []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            r, c = positions[i, j]
            if np.isnan(r): continue
            dr = r - gt_2d[i, j, 0]; dc = c - gt_2d[i, j, 1]
            errors.append(np.sqrt(dr**2 + dc**2))
    return np.array(errors)


def signed_bias(positions, gt_2d):
    dr_l, dc_l = [], []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            r, c = positions[i, j]
            if np.isnan(r): continue
            dr_l.append(r - gt_2d[i, j, 0]); dc_l.append(c - gt_2d[i, j, 1])
    return np.mean(dr_l), np.mean(dc_l), np.std(dr_l), np.std(dc_l)


print("Computing GT projections...")
gt_v2 = project_v2(marker_3d, vectors, DET_ROWS, DET_COLS)
gt_v3 = project_v3(marker_3d, vectors, DET_ROWS, DET_COLS)

# Compare V2 and V3 differences
diff = gt_v3 - gt_v2
print(f"\nV3 vs V2 formula difference:")
print(f"  max row diff: {np.abs(diff[:,:,0]).max():.4f}px")
print(f"  max col diff: {np.abs(diff[:,:,1]).max():.4f}px")
print(f"  mean row diff: {diff[:,:,0].mean():.4f}px")
print(f"  mean col diff: {diff[:,:,1].mean():.4f}px")
print(f"  RMS row diff: {np.sqrt((diff[:,:,0]**2).mean()):.4f}px")

# How do blob positions compare to V3?
print("\n[H3] Blob positions vs GT v3 (cross-product normal, (N-1)/2 center)")
e3 = rms_errors(blob_pos, gt_v3)
dr3, dc3, sr3, sc3 = signed_bias(blob_pos, gt_v3)
print(f"     mean dist={e3.mean():.4f}px  RMS={np.sqrt((e3**2).mean()):.4f}px")
print(f"     bias: dr={dr3:+.4f} (std={sr3:.4f})  dc={dc3:+.4f} (std={sc3:.4f})")

print("\n[H2] Blob positions vs GT v2 (beam_unit normal, (N-1)/2 center)")
e2 = rms_errors(blob_pos, gt_v2)
dr2, dc2, sr2, sc2 = signed_bias(blob_pos, gt_v2)
print(f"     mean dist={e2.mean():.4f}px  RMS={np.sqrt((e2**2).mean()):.4f}px")
print(f"     bias: dr={dr2:+.4f} (std={sr2:.4f})  dc={dc2:+.4f} (std={sc2:.4f})")


# ---- Gaussian PSF fit with V3 GT -----------------------------------------
def gaussian_fit_single(proj, r_exp, c_exp, window=20, max_offset=3.0):
    """Gaussian PSF fit on background-subtracted contrast."""
    det_rows, det_cols = proj.shape
    half = window // 2
    contrast = proj.astype(np.float64) - gaussian_filter(proj.astype(np.float64), sigma=30.0)
    r0 = int(round(r_exp)); c0 = int(round(c_exp))
    r_lo = max(0, r0 - half); r_hi = min(det_rows, r0 + half + 1)
    c_lo = max(0, c0 - half); c_hi = min(det_cols, c0 + half + 1)
    patch = contrast[r_lo:r_hi, c_lo:c_hi]
    nr, nc = patch.shape
    if nr < 5 or nc < 5 or patch.max() <= 0: return r_exp, c_exp
    r_init = float(r0 - r_lo); c_init = float(c0 - c_lo)
    rr, cc = np.mgrid[0:nr, 0:nc].astype(np.float64)
    rr_f = rr.ravel(); cc_f = cc.ravel(); data = patch.ravel()
    A0 = max(patch.max() - patch.mean(), 1e-6); bg0 = float(patch.mean())
    def residuals(p):
        r0p, c0p, A, sigma, bg = p
        return bg + A*np.exp(-((rr_f-r0p)**2+(cc_f-c0p)**2)/(2*sigma**2)) - data
    p0 = [r_init, c_init, A0, 4.0, bg0]
    lo = [max(0.0, r_init - max_offset), max(0.0, c_init - max_offset), 0, 0.5, -np.inf]
    hi = [min(nr-1.0, r_init + max_offset), min(nc-1.0, c_init + max_offset), np.inf, 12.0, np.inf]
    try:
        res = least_squares(residuals, p0, bounds=(lo, hi), method='trf', max_nfev=300)
        r_f = r_lo + res.x[0]; c_f = c_lo + res.x[1]
        if abs(r_f - r_exp) < max_offset + 1 and abs(c_f - c_exp) < max_offset + 1:
            return r_f, c_f
    except Exception: pass
    return r_exp, c_exp


print("\nGaussian PSF fit (seeded from GT v3, max_offset=3px, all 100 shots)...")
gfit_v3 = np.full((N_SHOTS, N_MARKERS, 2), np.nan)
for i in range(N_SHOTS):
    proj = sinogram[:, i, :]
    for j in range(N_MARKERS):
        r_exp, c_exp = gt_v3[i, j]
        if 0 <= r_exp < DET_ROWS and 0 <= c_exp < DET_COLS:
            gfit_v3[i, j] = gaussian_fit_single(proj, r_exp, c_exp, window=20, max_offset=3.0)

e_gfit_v3 = rms_errors(gfit_v3, gt_v3)
dr_g3, dc_g3, sr_g3, sc_g3 = signed_bias(gfit_v3, gt_v3)
n_valid = int((~np.isnan(gfit_v3[:, :, 0])).sum())
noise_ps = np.full(N_SHOTS, np.nan)
for i in range(N_SHOTS):
    esq = []
    for j in range(N_MARKERS):
        r, c = gfit_v3[i, j]
        if np.isnan(r): continue
        dr = r - gt_v3[i, j, 0]; dc = c - gt_v3[i, j, 1]
        esq.append(dr**2 + dc**2)
    if esq: noise_ps[i] = np.sqrt(np.mean(esq))
valid_ps = noise_ps[~np.isnan(noise_ps)]

print(f"  valid: {n_valid}/{N_SHOTS*N_MARKERS}")
print(f"  mean dist={e_gfit_v3.mean():.4f}px  RMS={np.sqrt((e_gfit_v3**2).mean()):.4f}px")
print(f"  bias: dr={dr_g3:+.4f} (std={sr_g3:.4f})  dc={dc_g3:+.4f} (std={sc_g3:.4f})")
print(f"  per-shot: mean={valid_ps.mean():.4f}px  max={valid_ps.max():.4f}px  median={np.median(valid_ps):.4f}px")

# Distribution
thresholds = [0.05, 0.10, 0.15, 0.20, 0.50, 1.0]
print(f"  Per-shot noise distribution:")
for t in thresholds:
    n_pass = int((valid_ps < t).sum())
    print(f"    < {t:.2f}px: {n_pass}/{len(valid_ps)} shots ({100*n_pass/len(valid_ps):.1f}%)")

if valid_ps.mean() < 0.15:
    print("  PASS: mean < 0.15px (CENT-005)")
else:
    print("  FAIL: mean >= 0.15px")

# Per-marker breakdown
print("\nPer-marker noise (Euclidean dist mean over all shots):")
for j in range(N_MARKERS):
    errs = []
    for i in range(N_SHOTS):
        r, c = gfit_v3[i, j]
        if np.isnan(r): continue
        dr = r - gt_v3[i, j, 0]; dc = c - gt_v3[i, j, 1]
        errs.append(np.sqrt(dr**2 + dc**2))
    m = marker_3d[j]
    print(f"  [{j}] ({m[0]:+.1f},{m[1]:+.1f},{m[2]:+.1f}): "
          f"mean={np.mean(errs):.4f}  std={np.std(errs):.4f}  n={len(errs)}")

print("\nDone.")
