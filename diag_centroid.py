"""
diag_centroid.py - Diagnose centroiding noise root cause.

Tests two hypotheses:
  H1: GT formula pixel offset is wrong (det_cols/2 vs (det_cols-1)/2)
  H2: Refinement algorithm is suboptimal (CoM vs Gaussian fit)

Runs on saved sinogram_100.h5.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

HDF5 = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
DET_ROWS = 512
DET_COLS = 512
DET_SPACING = 0.2

print("Loading sinogram_100.h5...")
with h5py.File(HDF5, 'r') as f:
    sinogram     = f['sinogram'][:]
    vectors      = f['shots/nominal/cone_vec'][:]
    gt_9dof      = f['shots/nominal/ground_truth_9dof'][:]
    blob_pos     = f['centroids/positions'][:]        # (100, 8, 2) current

with h5py.File(ROOT / 'data' / 'navy' / 'phantom.h5', 'r') as f:
    marker_3d = f['marker_positions'][:].astype(np.float64)  # (8, 3)

print(f"  sinogram: {sinogram.shape}  vectors: {vectors.shape}")
print(f"  markers: {marker_3d.shape}")
for i, m in enumerate(marker_3d):
    print(f"    [{i}]: ({m[0]:.3f}, {m[1]:.3f}, {m[2]:.3f})")

n_shots, n_markers = 100, 8


# ---- GT projection helpers --------------------------------------------------

def _project_markers(marker_3d, vectors, det_rows, det_cols, center_offset):
    """
    Project markers using pinhole formula.
    center_offset: 0.5*det_cols for current, 0.5*(det_cols-1) for ASTRA convention.
    Actually pass the CENTER pixel index: det_cols/2 vs (det_cols-1)/2.
    """
    n_shots   = len(vectors)
    n_markers = len(marker_3d)
    gt_2d = np.zeros((n_shots, n_markers, 2), dtype=np.float64)

    col_ctr = center_offset        # pixel index of detector center column
    row_ctr = center_offset        # same for rows (square detector)

    for i in range(n_shots):
        src     = vectors[i, 0:3]
        det_ctr = vectors[i, 3:6]
        u_vec   = vectors[i, 6:9]
        v_vec   = vectors[i, 9:12]
        det_spacing = np.linalg.norm(u_vec)
        u_hat = u_vec / det_spacing
        v_hat = v_vec / det_spacing
        beam_vec  = det_ctr - src
        beam_unit = beam_vec / np.linalg.norm(beam_vec)

        for j, m in enumerate(marker_3d):
            direction = m - src
            denom = np.dot(beam_unit, direction)
            if abs(denom) < 1e-10:
                gt_2d[i, j] = [row_ctr, col_ctr]
                continue
            t = np.dot(beam_unit, det_ctr - src) / denom
            P_det = src + t * direction
            delta = P_det - det_ctr
            col = np.dot(delta, u_hat) / det_spacing + col_ctr
            row = np.dot(delta, v_hat) / det_spacing + row_ctr
            gt_2d[i, j] = [row, col]

    return gt_2d


# Current formula uses det_cols/2 = 256.0
gt_v1 = _project_markers(marker_3d, vectors, DET_ROWS, DET_COLS, DET_COLS / 2.0)
# ASTRA convention uses (det_cols-1)/2 = 255.5
gt_v2 = _project_markers(marker_3d, vectors, DET_ROWS, DET_COLS, (DET_COLS - 1) / 2.0)


# ---- Analysis helpers -------------------------------------------------------

def rms_errors(positions, gt_2d):
    errors = []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            r, c = positions[i, j]
            if np.isnan(r): continue
            dr = r - gt_2d[i, j, 0]
            dc = c - gt_2d[i, j, 1]
            errors.append(np.sqrt(dr**2 + dc**2))
    return np.array(errors)


def signed_bias(positions, gt_2d):
    dr_list, dc_list = [], []
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            r, c = positions[i, j]
            if np.isnan(r): continue
            dr_list.append(r - gt_2d[i, j, 0])
            dc_list.append(c - gt_2d[i, j, 1])
    return (np.mean(dr_list), np.mean(dc_list),
            np.std(dr_list),  np.std(dc_list))


# ---- H1: Test GT pixel-offset convention ------------------------------------

print("\n[H1] Blob positions vs GT v1 (current: center=256.0)")
e1 = rms_errors(blob_pos, gt_v1)
dr1, dc1, sr1, sc1 = signed_bias(blob_pos, gt_v1)
print(f"     mean dist={e1.mean():.4f}px  RMS={np.sqrt((e1**2).mean()):.4f}px")
print(f"     bias: dr={dr1:+.4f} (std={sr1:.4f})  dc={dc1:+.4f} (std={sc1:.4f})")

print("\n[H1] Blob positions vs GT v2 (ASTRA: center=255.5)")
e2 = rms_errors(blob_pos, gt_v2)
dr2, dc2, sr2, sc2 = signed_bias(blob_pos, gt_v2)
print(f"     mean dist={e2.mean():.4f}px  RMS={np.sqrt((e2**2).mean()):.4f}px")
print(f"     bias: dr={dr2:+.4f} (std={sr2:.4f})  dc={dc2:+.4f} (std={sc2:.4f})")

print(f"\n     GT convention change reduces mean dist by {e1.mean()-e2.mean():.4f}px")


# ---- H2: Gaussian PSF fit refinement ----------------------------------------

def gaussian_fit_refine(projection, expected_pos, window=20):
    """
    2D symmetric Gaussian + background fit on background-subtracted contrast.
    Seeded from expected_pos (GT or SDSG-converged estimate).
    Returns refined positions or expected_pos on failure.
    """
    det_rows, det_cols = projection.shape
    half = window // 2
    proj = projection.astype(np.float64)
    contrast = proj - gaussian_filter(proj, sigma=30.0)

    n_m = len(expected_pos)
    refined = np.full((n_m, 2), np.nan, dtype=np.float64)

    for j, (r_exp, c_exp) in enumerate(expected_pos):
        if np.isnan(r_exp): continue

        r0 = int(round(r_exp)); c0 = int(round(c_exp))
        r_lo = max(0, r0 - half); r_hi = min(det_rows, r0 + half + 1)
        c_lo = max(0, c0 - half); c_hi = min(det_cols, c0 + half + 1)
        patch = contrast[r_lo:r_hi, c_lo:c_hi]
        nr, nc = patch.shape

        if nr < 5 or nc < 5 or patch.max() <= 0:
            refined[j] = [r_exp, c_exp]; continue

        r_init = float(r0 - r_lo)
        c_init = float(c0 - c_lo)
        rr, cc  = np.mgrid[0:nr, 0:nc].astype(np.float64)
        rr_f = rr.ravel(); cc_f = cc.ravel(); data = patch.ravel()

        A0   = max(patch.max() - patch.mean(), 1e-6)
        bg0  = float(patch.mean())

        def residuals(p):
            r0p, c0p, A, sigma, bg = p
            model = bg + A * np.exp(
                -((rr_f - r0p)**2 + (cc_f - c0p)**2) / (2.0 * sigma**2))
            return model - data

        p0 = [r_init, c_init, A0, 4.0, bg0]
        lo = [max(0.0, r_init - 6), max(0.0, c_init - 6), 0, 0.5, -np.inf]
        hi = [min(nr - 1.0, r_init + 6), min(nc - 1.0, c_init + 6), np.inf, 12.0, np.inf]

        try:
            res = least_squares(residuals, p0, bounds=(lo, hi),
                                method='trf', max_nfev=300)
            r_fit = r_lo + res.x[0]
            c_fit = c_lo + res.x[1]
            if abs(r_fit - r_exp) < 8 and abs(c_fit - c_exp) < 8:
                refined[j] = [r_fit, c_fit]
            else:
                refined[j] = [r_exp, c_exp]
        except Exception:
            refined[j] = [r_exp, c_exp]

    return refined


print("\n[H2] Gaussian PSF fit (seeded from GT v2, all 100 shots)...")
gfit = np.full((n_shots, n_markers, 2), np.nan, dtype=np.float64)
for i in range(n_shots):
    gfit[i] = gaussian_fit_refine(sinogram[:, i, :], gt_v2[i], window=20)

e_gfit = rms_errors(gfit, gt_v2)
dr_g, dc_g, sr_g, sc_g = signed_bias(gfit, gt_v2)
n_valid = int((~np.isnan(gfit[:, :, 0])).sum())
print(f"     valid: {n_valid}/{n_shots*n_markers}")
print(f"     mean dist={e_gfit.mean():.4f}px  RMS={np.sqrt((e_gfit**2).mean()):.4f}px")
print(f"     bias: dr={dr_g:+.4f} (std={sr_g:.4f})  dc={dc_g:+.4f} (std={sc_g:.4f})")

# per-shot breakdown
noise_ps = np.full(n_shots, np.nan)
for i in range(n_shots):
    esq = []
    for j in range(n_markers):
        r, c = gfit[i, j]
        if np.isnan(r): continue
        dr = r - gt_v2[i, j, 0]; dc = c - gt_v2[i, j, 1]
        esq.append(dr**2 + dc**2)
    if esq: noise_ps[i] = np.sqrt(np.mean(esq))

valid_ps = noise_ps[~np.isnan(noise_ps)]
print(f"     per-shot: mean={valid_ps.mean():.4f}px  max={valid_ps.max():.4f}px")

if valid_ps.mean() < 0.15:
    print("     PASS: mean < 0.15px  (CENT-005)")
else:
    print("     FAIL: mean >= 0.15px")
    worst_idx = np.argsort(valid_ps)[-5:]
    print("     5 worst shots (noise, px):")
    for k in worst_idx:
        print(f"       shot {k}: {valid_ps[k]:.4f}px")

print("\nDone.")
