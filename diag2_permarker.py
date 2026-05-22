"""
diag2_permarker.py - Per-marker signed bias analysis.

Checks if centroiding errors are per-marker systematic (GT formula error)
or random per-shot (Poisson noise).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

HDF5   = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
PHANTOM = ROOT / 'data' / 'navy' / 'phantom.h5'

with h5py.File(HDF5, 'r') as f:
    sinogram = f['sinogram'][:]
    vectors  = f['shots/nominal/cone_vec'][:]

with h5py.File(PHANTOM, 'r') as f:
    marker_3d = f['marker_positions'][:].astype(np.float64)
    voxel_size = float(f.attrs['voxel_size_mm'])

print(f"voxel_size = {voxel_size}mm")

DET_ROWS, N_SHOTS, DET_COLS = sinogram.shape
DET_SPACING = 0.2
N_MARKERS = len(marker_3d)


def project_gt(marker_3d, vectors, det_rows, det_cols, use_astra_offset=True):
    """GT projection. use_astra_offset=True uses (N-1)/2 pixel offset."""
    offset = (det_cols - 1) / 2.0 if use_astra_offset else det_cols / 2.0
    n_shots = len(vectors)
    n_m = len(marker_3d)
    gt_2d = np.zeros((n_shots, n_m, 2), dtype=np.float64)
    for i in range(n_shots):
        src    = vectors[i, 0:3]; det_ctr = vectors[i, 3:6]
        u_vec  = vectors[i, 6:9]; v_vec   = vectors[i, 9:12]
        ds = np.linalg.norm(u_vec)
        u_hat = u_vec / ds; v_hat = v_vec / ds
        beam_unit = (det_ctr - src) / np.linalg.norm(det_ctr - src)
        for j, m in enumerate(marker_3d):
            d = m - src
            denom = np.dot(beam_unit, d)
            if abs(denom) < 1e-10:
                gt_2d[i, j] = [offset, offset]; continue
            t = np.dot(beam_unit, det_ctr - src) / denom
            P = src + t * d
            delta = P - det_ctr
            col = np.dot(delta, u_hat) / ds + offset
            row = np.dot(delta, v_hat) / ds + offset
            gt_2d[i, j] = [row, col]
    return gt_2d


gt_v2 = project_gt(marker_3d, vectors, DET_ROWS, DET_COLS, use_astra_offset=True)


def gaussian_fit_single(proj, r_exp, c_exp, window=20):
    """Gaussian PSF fit on background-subtracted contrast. Returns (r_fit, c_fit)."""
    det_rows, det_cols = proj.shape
    half = window // 2
    contrast = proj.astype(np.float64) - gaussian_filter(proj.astype(np.float64), sigma=30.0)

    r0 = int(round(r_exp)); c0 = int(round(c_exp))
    r_lo = max(0, r0 - half); r_hi = min(det_rows, r0 + half + 1)
    c_lo = max(0, c0 - half); c_hi = min(det_cols, c0 + half + 1)
    patch = contrast[r_lo:r_hi, c_lo:c_hi]
    nr, nc = patch.shape

    if nr < 5 or nc < 5 or patch.max() <= 0:
        return r_exp, c_exp

    r_init = float(r0 - r_lo); c_init = float(c0 - c_lo)
    rr, cc = np.mgrid[0:nr, 0:nc].astype(np.float64)
    rr_f = rr.ravel(); cc_f = cc.ravel(); data = patch.ravel()
    A0 = max(patch.max() - patch.mean(), 1e-6)
    bg0 = float(patch.mean())

    def residuals(p):
        r0p, c0p, A, sigma, bg = p
        return bg + A * np.exp(-((rr_f-r0p)**2+(cc_f-c0p)**2)/(2*sigma**2)) - data

    p0 = [r_init, c_init, A0, 4.0, bg0]
    lo = [max(0.0, r_init-6), max(0.0, c_init-6), 0, 0.5, -np.inf]
    hi = [min(nr-1.0,r_init+6), min(nc-1.0,c_init+6), np.inf, 12.0, np.inf]
    try:
        res = least_squares(residuals, p0, bounds=(lo,hi), method='trf', max_nfev=300)
        r_f = r_lo + res.x[0]; c_f = c_lo + res.x[1]
        if abs(r_f - r_exp) < 8 and abs(c_f - c_exp) < 8:
            return r_f, c_f
    except Exception:
        pass
    return r_exp, c_exp


# ---- Per-marker bias across all shots ----------------------------------------
print("\nRunning Gaussian fit on all shots...")
fit_pos = np.full((N_SHOTS, N_MARKERS, 2), np.nan)
for i in range(N_SHOTS):
    proj = sinogram[:, i, :]
    for j in range(N_MARKERS):
        r_exp, c_exp = gt_v2[i, j]
        if 0 <= r_exp < DET_ROWS and 0 <= c_exp < DET_COLS:
            fit_pos[i, j] = gaussian_fit_single(proj, r_exp, c_exp)

print("\nPer-marker signed bias (dr, dc) averaged across all shots:")
print(f"{'Marker':>6}  {'dr_mean':>8}  {'dr_std':>8}  {'dc_mean':>8}  {'dc_std':>8}  {'dist_mean':>10}")
per_marker_bias = []
for j in range(N_MARKERS):
    dr_list, dc_list = [], []
    for i in range(N_SHOTS):
        r_f, c_f = fit_pos[i, j]
        if np.isnan(r_f): continue
        dr_list.append(r_f - gt_v2[i, j, 0])
        dc_list.append(c_f - gt_v2[i, j, 1])
    if dr_list:
        dr_m, dc_m = np.mean(dr_list), np.mean(dc_list)
        dr_s, dc_s = np.std(dr_list), np.std(dc_list)
        dist_m = np.mean(np.sqrt(np.array(dr_list)**2 + np.array(dc_list)**2))
        per_marker_bias.append((dr_m, dc_m, dr_s, dc_s))
        m = marker_3d[j]
        print(f"  [{j}] ({m[0]:+.1f},{m[1]:+.1f},{m[2]:+.1f}):"
              f"  dr={dr_m:+.4f}(std={dr_s:.4f})  dc={dc_m:+.4f}(std={dc_s:.4f})  dist={dist_m:.4f}")

# ---- Per-shot bias across all markers ---------------------------------------
print("\nPer-shot noise (Euclidean dist mean over all 8 markers):")
worst_shots = []
for i in range(N_SHOTS):
    dists = []
    for j in range(N_MARKERS):
        r_f, c_f = fit_pos[i, j]
        if np.isnan(r_f): continue
        dr = r_f - gt_v2[i, j, 0]; dc = c_f - gt_v2[i, j, 1]
        dists.append(np.sqrt(dr**2 + dc**2))
    if dists:
        mean_d = np.mean(dists)
        if mean_d > 1.0:
            worst_shots.append((mean_d, i))

worst_shots.sort(reverse=True)
print(f"  Shots with mean dist > 1.0px: {len(worst_shots)}")
for d, i in worst_shots[:10]:
    # Compute source position for this shot
    src = vectors[i, 0:3]
    src_norm = np.linalg.norm(src)
    cos_theta = src[2] / src_norm
    print(f"    shot {i:3d}: mean_dist={d:.4f}px  cos_theta={cos_theta:.4f}  src=({src[0]:+.1f},{src[1]:+.1f},{src[2]:+.1f})")

# ---- Check if errors are larger for edge markers ----------------------------
print("\nMarker distance from phantom center vs mean error:")
for j in range(N_MARKERS):
    m = marker_3d[j]
    dist3d = np.linalg.norm(m)
    dr_list, dc_list = [], []
    for i in range(N_SHOTS):
        r_f, c_f = fit_pos[i, j]
        if np.isnan(r_f): continue
        dr_list.append(r_f - gt_v2[i, j, 0])
        dc_list.append(c_f - gt_v2[i, j, 1])
    if dr_list:
        err_m = np.mean(np.sqrt(np.array(dr_list)**2 + np.array(dc_list)**2))
        print(f"  [{j}] |m|={dist3d:.2f}mm  mean_err={err_m:.4f}px")

# ---- Check shots 0, 1, 2 geometry -------------------------------------------
print("\nGeometry for worst shots:")
for _, i in worst_shots[:5]:
    src = vectors[i, 0:3]
    src_norm = np.linalg.norm(src)
    cos_theta = src[2] / src_norm
    print(f"  shot {i}: src=({src[0]:+6.1f},{src[1]:+6.1f},{src[2]:+6.1f})  |src|={src_norm:.1f}  cos_theta={cos_theta:.4f}")
    # Show per-marker error for this shot
    for j in range(N_MARKERS):
        r_f, c_f = fit_pos[i, j]
        r_exp, c_exp = gt_v2[i, j]
        if np.isnan(r_f):
            print(f"    marker {j}: NaN")
        else:
            dr = r_f - r_exp; dc = c_f - c_exp
            print(f"    marker {j}: err=({dr:+.3f},{dc:+.3f})  |err|={np.sqrt(dr**2+dc**2):.3f}px  "
                  f"expected=({r_exp:.1f},{c_exp:.1f})")

print("\nDone.")
