"""
analysis/figures_navy.py — All Navy SBIR deliverable figures and tables.

Figures:
  NV-FIG-01  fig01_residual_histogram.png  — Solver residual histogram
  NV-FIG-02  fig02_recon_comparison.png    — GT vs FBP vs mART (axial slice)
  NV-FIG-03  fig03_accuracy_vs_shots.png   — Mean residual vs N_shots
  NV-FIG-04  fig04_cnr_vs_crack.png        — CNR vs crack width

Tables:
  NV-TAB-01  tab01_geometry_accuracy.csv   — Geometry accuracy: nominal + stress
  NV-TAB-02  tab02_image_quality.csv       — Image quality: FBP + mART

Usage:
  python analysis/figures_navy.py

Figures that depend on not-yet-computed data will be skipped with a warning.
"""

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

ROOT      = Path(__file__).resolve().parent.parent
SINO_PATH = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
FIG_DIR   = ROOT / 'figures' / 'navy'

MU_STEEL  = 0.1149   # mm^-1 at 200keV — colormap max for recon panels

# ─────────────────────────────────────────────────────────────────────────────
# NV-FIG-01 — Residual Histogram
# ─────────────────────────────────────────────────────────────────────────────

def fig01_residual_histogram() -> None:
    """
    Solver residual histogram from sinogram_100.h5.
    Reads results/residuals/per_marker (100, 8), flattens, drops NaN.
    """
    print("[NV-FIG-01] Residual histogram...")

    with h5py.File(SINO_PATH, 'r') as f:
        if 'results/residuals/per_marker' not in f:
            print("  SKIP: results/residuals/per_marker not found in HDF5")
            return
        per_marker = f['results/residuals/per_marker'][:]   # (100, 8)
        mean_res   = float(f['results/residuals'].attrs.get('mean', np.nanmean(per_marker)))
        p95_res    = float(f['results/residuals'].attrs.get('p95',
                           np.nanpercentile(per_marker[~np.isnan(per_marker)], 95)))

    flat = per_marker.flatten()
    flat = flat[~np.isnan(flat)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(flat, bins=50, range=(0.0, 1.0), color='steelblue', edgecolor='white',
            linewidth=0.4, density=False)

    ax.axvline(0.2, color='red', linestyle='--', linewidth=1.5, label='0.2 px target')
    ax.axvline(mean_res, color='orange', linestyle='-', linewidth=1.5,
               label=f'Mean = {mean_res:.4f} px')

    ax.set_xlabel('Reprojection Residual (pixels)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('SDSG Solver — Per-Marker Reprojection Residuals\n(100 shots × 8 markers)',
                 fontsize=13)
    ax.legend(fontsize=12)
    ax.annotate(f'Mean = {mean_res:.4f} px\nP95  = {p95_res:.4f} px',
                xy=(0.97, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.tick_params(labelsize=11)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()

    out = FIG_DIR / 'fig01_residual_histogram.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# NV-FIG-02 — GT vs FBP vs mART (3-panel)
# ─────────────────────────────────────────────────────────────────────────────

def fig02_recon_comparison() -> None:
    """
    Axial slice (Y=-0.5mm → iy=120) comparing GT phantom, FBP, and mART.
    """
    print("[NV-FIG-02] Reconstruction comparison...")

    with h5py.File(SINO_PATH, 'r') as f:
        if 'recon/vol_fbp_inpainted' not in f or 'recon/vol_mart_inpainted' not in f:
            print("  SKIP: recon results not found — run recon/run_navy_recon.py first")
            return
        vol_fbp   = f['recon/vol_fbp_inpainted'][:]   # (250,250,250)
        vol_mart  = f['recon/vol_mart_inpainted'][:]
        vol_ref   = f['recon/phantom_inpainted'][:]

    iy = 120   # crack center Y=-0.5mm

    panels = [
        (vol_ref[:, iy, :],  'A — Ground Truth', False),
        (vol_fbp[:, iy, :],  'B — FBP (FDK, non-circular orbit)', True),
        (vol_mart[:, iy, :], 'C — mART (50 iter)', False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vmin, vmax = 0.0, MU_STEEL

    for ax, (img, title, note_fbp) in zip(axes, panels):
        im = ax.imshow(img.T, origin='lower', cmap='gray',
                       vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (voxel)', fontsize=10)
        ax.set_ylabel('Z (voxel)', fontsize=10)
        ax.tick_params(labelsize=9)

        # Arrow pointing at crack location (iy=120 is the slice, crack runs in X)
        # Crack appears as a horizontal band; arrow from top at approximate X center
        mid_x = img.shape[0] // 2
        ax.annotate('', xy=(mid_x, 120), xytext=(mid_x, 145),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.text(mid_x + 5, 140, 'crack', color='red', fontsize=8)

        if note_fbp:
            ax.text(0.02, 0.02, 'Non-circular orbit -- streaks expected',
                    transform=ax.transAxes, fontsize=8, color='yellow',
                    bbox=dict(facecolor='black', alpha=0.5))

        # 2mm scale bar (20 voxels at 0.1mm/vox) in bottom-right
        bar_len = 20   # voxels = 2mm
        x_bar0 = img.shape[0] - 30
        z_bar  = 15
        ax.plot([x_bar0, x_bar0 + bar_len], [z_bar, z_bar], 'w-', linewidth=2)
        ax.text(x_bar0 + bar_len // 2, z_bar + 4, '2 mm',
                color='white', fontsize=8, ha='center')

    plt.colorbar(im, ax=axes[-1], label='mu (mm^-1)', fraction=0.046, pad=0.04)
    fig.suptitle('SDSG Navy — Axial Slice at Crack Center (Y = −0.5 mm)', fontsize=13)
    fig.tight_layout()

    out = FIG_DIR / 'fig02_recon_comparison.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# NV-FIG-03 — Accuracy vs N_shots
# ─────────────────────────────────────────────────────────────────────────────

def fig03_accuracy_vs_shots() -> None:
    """
    Mean residual ± 1 std as a function of N_shots (20, 50, 100, 200).
    Reads variant HDF5 files for 20/50/200 and sinogram_100.h5 for 100.
    """
    print("[NV-FIG-03] Accuracy vs N_shots...")

    n_list     = [20, 50, 100, 200]
    means      = []
    stds       = []

    for n in n_list:
        if n == 100:
            path = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
        else:
            path = ROOT / 'data' / 'navy' / f'sinogram_{n}.h5'

        if not path.exists():
            print(f"  SKIP: {path} not found")
            return

        with h5py.File(path, 'r') as f:
            if 'results/residuals/per_shot' not in f:
                print(f"  SKIP: results not in {path}")
                return
            per_shot = f['results/residuals/per_shot'][:]

        valid = per_shot[~np.isnan(per_shot)]
        means.append(float(np.mean(valid)) if len(valid) else np.nan)
        stds.append(float(np.std(valid))  if len(valid) else 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(n_list, means, yerr=stds, fmt='o-', capsize=5,
                color='steelblue', linewidth=2, markersize=7, label='Mean ± 1 std')
    ax.axhline(0.2, color='red', linestyle='--', linewidth=1.5, label='0.2 px target')

    ax.set_xlabel('Number of Shots', fontsize=13)
    ax.set_ylabel('Mean Reprojection Residual (pixels)', fontsize=13)
    ax.set_title('SDSG Geometry Recovery — Accuracy vs. Shot Count', fontsize=13)
    ax.legend(fontsize=12)
    ax.set_xscale('log')
    ax.set_xticks(n_list)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()

    out = FIG_DIR / 'fig03_accuracy_vs_shots.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# NV-FIG-04 — CNR vs Crack Width
# ─────────────────────────────────────────────────────────────────────────────

def fig04_cnr_vs_crack() -> None:
    """
    CNR vs crack width for FBP and mART, with Rose criterion line at CNR=4.
    """
    print("[NV-FIG-04] CNR vs crack width...")

    with h5py.File(SINO_PATH, 'r') as f:
        if 'recon/metrics' not in f:
            print("  SKIP: recon/metrics not found — run recon/run_navy_recon.py first")
            return
        widths   = list(f['recon/metrics/crack_widths_mm'][:])
        cnr_fbp  = list(f['recon/metrics/cnr_fbp'][:])
        cnr_mart = list(f['recon/metrics/cnr_mart'][:])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(widths, cnr_fbp,  'b--o', linewidth=2, markersize=7, label='FBP (baseline)')
    ax.plot(widths, cnr_mart, 'r-o',  linewidth=2, markersize=7, label='mART (50 iter)')
    ax.axhline(4.0, color='green', linestyle='--', linewidth=1.5, label='Rose criterion (CNR=4)')

    ax.set_xlabel('Crack Width (mm)', fontsize=13)
    ax.set_ylabel('CNR', fontsize=13)
    ax.set_title('SDSG Navy — CNR vs. Crack Width', fontsize=13)
    ax.legend(fontsize=12)
    ax.set_xticks(widths)
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()

    out = FIG_DIR / 'fig04_cnr_vs_crack.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# NV-TAB-01 — Geometry Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def tab01_geometry_accuracy() -> None:
    """
    Two-row table: nominal (sigma_s=2mm) and stress (sigma_s=10mm) conditions.
    """
    print("[NV-TAB-01] Geometry accuracy table...")

    rows = []

    # Nominal: from sinogram_100.h5
    nom_path = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
    if not nom_path.exists():
        print("  SKIP: sinogram_100.h5 not found")
        return

    with h5py.File(nom_path, 'r') as f:
        if 'results/residuals/per_shot' not in f:
            print("  SKIP: nominal solver results not found")
            return
        per_shot   = f['results/residuals/per_shot'][:]
        pos_err    = f['results/position_error_mm'][:]
        ang_err    = f['results/angular_error_deg'][:]

    valid = per_shot[~np.isnan(per_shot)]
    rows.append({
        'condition':      'Nominal (sigma_s=2mm, sigma_theta=1deg)',
        'mean_pos_err_mm': f'{np.nanmean(pos_err):.3f}',
        'std_pos_err_mm':  f'{np.nanstd(pos_err):.3f}',
        'mean_ang_err_deg': f'{np.nanmean(ang_err):.3f}',
        'mean_residual_px': f'{np.mean(valid):.4f}',
        'p95_residual_px':  f'{np.percentile(valid, 95):.4f}',
    })

    # Stress: from sinogram_100_stress.h5
    stress_path = ROOT / 'data' / 'navy' / 'sinogram_100_stress.h5'
    if stress_path.exists():
        with h5py.File(stress_path, 'r') as f:
            if 'results/residuals/per_shot' in f:
                per_shot_s = f['results/residuals/per_shot'][:]
                pos_err_s  = f['results/position_error_mm'][:]
                ang_err_s  = f['results/angular_error_deg'][:]
        valid_s = per_shot_s[~np.isnan(per_shot_s)]
        rows.append({
            'condition':      'Stress (sigma_s=10mm, sigma_theta=5deg)',
            'mean_pos_err_mm': f'{np.nanmean(pos_err_s):.3f}',
            'std_pos_err_mm':  f'{np.nanstd(pos_err_s):.3f}',
            'mean_ang_err_deg': f'{np.nanmean(ang_err_s):.3f}',
            'mean_residual_px': f'{np.mean(valid_s):.4f}',
            'p95_residual_px':  f'{np.percentile(valid_s, 95):.4f}',
        })
    else:
        print("  NOTE: stress results not yet available — table will have 1 row")

    out = FIG_DIR / 'tab01_geometry_accuracy.csv'
    fieldnames = ['condition', 'mean_pos_err_mm', 'std_pos_err_mm',
                  'mean_ang_err_deg', 'mean_residual_px', 'p95_residual_px']
    with open(out, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# NV-TAB-02 — Image Quality
# ─────────────────────────────────────────────────────────────────────────────

def tab02_image_quality() -> None:
    """
    Two-row table: FBP and mART image quality metrics.
    """
    print("[NV-TAB-02] Image quality table...")

    with h5py.File(SINO_PATH, 'r') as f:
        if 'recon/metrics' not in f:
            print("  SKIP: recon/metrics not found — run recon/run_navy_recon.py first")
            return
        mkeys = list(f['recon/metrics'].keys())
        m = {}
        for k in mkeys:
            try:
                m[k] = f[f'recon/metrics/{k}'][()]
            except Exception:
                pass

    rows = [
        {
            'method': 'FBP (FDK_CUDA)',
            'SSIM':        f"{m.get('ssim_fbp', float('nan')):.4f}",
            'PSNR_dB':     f"{m.get('psnr_fbp', float('nan')):.2f}",
            'CNR_0.4mm':   f"{m.get('cnr_fbp', [float('nan')])[0]:.2f}",
            'CNR_0.8mm':   f"{m.get('cnr_fbp', [float('nan'), float('nan')])[1]:.2f}",
            'CNR_1.6mm':   f"{m.get('cnr_fbp', [float('nan')]*3)[2]:.2f}",
        },
        {
            'method': 'mART (50 iter)',
            'SSIM':        f"{m.get('ssim_mart', float('nan')):.4f}",
            'PSNR_dB':     f"{m.get('psnr_mart', float('nan')):.2f}",
            'CNR_0.4mm':   f"{m.get('cnr_mart', [float('nan')])[0]:.2f}",
            'CNR_0.8mm':   f"{m.get('cnr_mart', [float('nan'), float('nan')])[1]:.2f}",
            'CNR_1.6mm':   f"{m.get('cnr_mart', [float('nan')]*3)[2]:.2f}",
        },
    ]

    out = FIG_DIR / 'tab02_image_quality.csv'
    fieldnames = ['method', 'SSIM', 'PSNR_dB', 'CNR_0.4mm', 'CNR_0.8mm', 'CNR_1.6mm']
    with open(out, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))

    print('=' * 60)
    print('Navy figures + tables generator')
    print('=' * 60)

    fig01_residual_histogram()
    fig02_recon_comparison()
    fig03_accuracy_vs_shots()
    fig04_cnr_vs_crack()
    tab01_geometry_accuracy()
    tab02_image_quality()

    print('\nDone. Check figures/navy/ for outputs.')


if __name__ == '__main__':
    main()
