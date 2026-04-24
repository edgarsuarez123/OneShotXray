"""
analysis/figures_nih.py -- All NIH proposal figures (300 DPI) to figures/nih/

Figures:
  NIH-AIM1-03: aim1_fig01_constellation_optimization.png  -- grouped bar chart
  NIH-AIM1-04: aim1_fig02_mart_convergence.png            -- mART convergence curves
  NIH-AIM2-01: aim2_fig01_hemorrhage_detection.png        -- 3-panel GT/FBP/mART
  NIH-AIM2-02: aim2_fig02_cnr_vs_lesion.png               -- CNR vs lesion size
  NIH-AIM2-03: aim2_fig03_arc_comparison.png              -- 4-panel arc comparison
  NIH-AIM2-04: aim2_fig04_image_quality.csv               -- summary table
  NIH-AIM2-05: aim2_fig05_simulated_roc.png               -- ROC curve + AUC
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR  = ROOT / 'figures' / 'nih'
DATA_DIR = ROOT / 'data' / 'nih'

DPI = 300
NIH_GRID       = (200, 160, 140)
NIH_VOXEL_SIZE = 1.0

# Hemorrhage at (58, 20, 0)mm -> voxel (158, 100, 70)
HEM_VX = int(58 / NIH_VOXEL_SIZE + NIH_GRID[0] / 2)  # 158
HEM_VY = int(20 / NIH_VOXEL_SIZE + NIH_GRID[1] / 2)  # 100
HEM_VZ = int(0  / NIH_VOXEL_SIZE + NIH_GRID[2] / 2)  # 70

# Soft tissue window: brain=0.021, hemorrhage=0.023, window +/-0.015
SOFT_WC = 0.021
SOFT_WW = 0.030
VMIN = SOFT_WC - SOFT_WW / 2.0
VMAX = SOFT_WC + SOFT_WW / 2.0


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ===========================================================================
# NIH-AIM1-03: Constellation optimization grouped bar chart
# ===========================================================================

def fig_constellation_optimization() -> None:
    path = DATA_DIR / 'aim1_constellation_grid.h5'
    if not path.exists():
        print(f'  SKIP NIH-AIM1-03: {path} not found')
        return

    with h5py.File(path, 'r') as f:
        arc_labels  = [s.decode() for s in f['arc_labels'][:]]
        marker_ns   = f['marker_ns'][:].tolist()
        residuals   = f['residuals_px'][:].tolist()

    # Group by arc
    arcs = ['restricted_180', 'full_360']
    arc_names = {'restricted_180': '180 deg Restricted', 'full_360': 'Full 360 deg'}
    marker_counts = sorted(set(marker_ns))
    x = np.arange(len(marker_counts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ['#2166ac', '#d6604d']
    bars = []
    for k, arc in enumerate(arcs):
        vals = [
            residuals[i] for i, (al, mn) in enumerate(zip(arc_labels, marker_ns))
            if al == arc
        ]
        b = ax.bar(x + k * width - width / 2, vals, width * 0.9,
                   label=arc_names[arc], color=colors[k], alpha=0.85, edgecolor='k', linewidth=0.5)
        bars.append(b)
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2., rect.get_height() + 0.0005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.axhline(0.3, color='red', linestyle='--', linewidth=1.2, label='Gate: 0.3 px')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n} Markers' for n in marker_counts])
    ax.set_ylabel('Mean Solver Residual (px)')
    ax.set_title('NIH Aim 1 -- Constellation Optimization\nSDSG Solver Residual vs Marker Count and Arc', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.35)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'aim1_fig01_constellation_optimization.png')


# ===========================================================================
# NIH-AIM1-04: mART convergence curves
# ===========================================================================

def fig_mart_convergence() -> None:
    path = DATA_DIR / 'aim1_mart_sweep.h5'
    if not path.exists():
        print(f'  SKIP NIH-AIM1-04: {path} not found')
        return

    with h5py.File(path, 'r') as f:
        n_iters   = f['n_iter'][:]
        relaxs    = f['relaxation'][:]
        conv_arr  = f['convergence'][:]
        cnr_arr   = f['cnr_5mm'][:]

    # Plot convergence for the 3 relaxation values at n_iter=100 (most iterations)
    target_niters = 100
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ['#1b7837', '#762a83', '#e08214']
    relax_vals = [0.5, 1.0, 2.0]
    for k, relax in enumerate(relax_vals):
        idx = [i for i, (n, r) in enumerate(zip(n_iters, relaxs))
               if n == target_niters and abs(r - relax) < 0.01]
        if not idx:
            continue
        i = idx[0]
        curve = conv_arr[i]
        valid_len = int(target_niters)
        iters = np.arange(1, valid_len + 1)
        ax.semilogy(iters, curve[:valid_len], color=colors[k], linewidth=1.5,
                    label=f'lambda={relax:.1f}  (CNR={cnr_arr[i]:.2f})')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('||correction|| / ||x|| (log scale)')
    ax.set_title('NIH Aim 1 -- mART Convergence\nRestricted 180 deg Arc, 8 Markers', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    _save(fig, 'aim1_fig02_mart_convergence.png')


# ===========================================================================
# NIH-AIM2-01: Hemorrhage detection -- 3-panel GT / FBP / mART
# ===========================================================================

def _axial_slice(vol: np.ndarray, iz: int, pad: int = 60) -> np.ndarray:
    """Axial slice through Z=iz, cropped around hemorrhage in XY plane."""
    sl = vol[:, :, iz]  # (nx, ny) -> show as (Y, X) with imshow
    # Crop around hemorrhage voxel
    r0 = max(0, HEM_VY - pad)
    r1 = min(sl.shape[1], HEM_VY + pad)
    c0 = max(0, HEM_VX - pad)
    c1 = min(sl.shape[0], HEM_VX + pad)
    return sl[c0:c1, r0:r1].T   # (ny_crop, nx_crop) for imshow (row=Y, col=X)


def fig_hemorrhage_detection() -> None:
    phantom_path = DATA_DIR / 'phantom_lesion_5mm.h5'
    fbp_path     = DATA_DIR / 'recon_restricted_fbp.h5'
    mart_path    = DATA_DIR / 'recon_restricted_mart.h5'

    for p in [phantom_path, fbp_path, mart_path]:
        if not p.exists():
            print(f'  SKIP NIH-AIM2-01: {p} not found')
            return

    with h5py.File(phantom_path, 'r') as f:
        gt_vol = f['volume'][:].astype(np.float32)
    with h5py.File(fbp_path, 'r') as f:
        fbp_vol = f['volume'][:].astype(np.float32)
    with h5py.File(mart_path, 'r') as f:
        mart_vol = f['volume'][:].astype(np.float32)

    iz = HEM_VZ
    panels = [
        (_axial_slice(gt_vol, iz),   'Ground Truth', 'Phantom'),
        (_axial_slice(fbp_vol, iz),  'FBP (FDK)',    '180 deg Restricted Arc'),
        (_axial_slice(mart_vol, iz), 'mART (25 iter)', '180 deg Restricted Arc'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (sl, title, subtitle) in zip(axes, panels):
        im = ax.imshow(sl, cmap='gray', vmin=VMIN, vmax=VMAX, origin='upper', aspect='equal')
        ax.set_title(f'{title}\n{subtitle}', fontsize=9)
        ax.set_xlabel('X (mm)', fontsize=8)
        ax.set_ylabel('Y (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        # Mark hemorrhage center
        hem_local_x = HEM_VX - max(0, HEM_VX - 60)
        hem_local_y = HEM_VY - max(0, HEM_VY - 60)
        ax.plot(hem_local_x, hem_local_y, 'r+', markersize=8, markeredgewidth=1.5,
                label='Hemorrhage')

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Attenuation (mm^-1)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('NIH Aim 2 -- Hemorrhage Detection\nSoft Tissue Window: c=0.021, w=0.030 mm^-1',
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, 'aim2_fig01_hemorrhage_detection.png')


# ===========================================================================
# NIH-AIM2-02: CNR vs lesion size
# ===========================================================================

def fig_cnr_vs_lesion() -> None:
    path = DATA_DIR / 'recon_lesion_sweep.h5'
    if not path.exists():
        print(f'  SKIP NIH-AIM2-02: {path} not found')
        return

    with h5py.File(path, 'r') as f:
        lesion_mm = f['lesion_mm'][:]
        arcs      = [s.decode() for s in f['arc'][:]]
        methods   = [s.decode() for s in f['method'][:]]
        cnr       = f['cnr'][:]

    configs = [
        ('restricted', 'fbp',  '#d6604d', '--', 'FBP Restricted 180 deg'),
        ('restricted', 'mart', '#2166ac', '-',  'mART Restricted 180 deg'),
        ('full360',    'fbp',  '#f4a582', '--', 'FBP Full 360 deg'),
        ('full360',    'mart', '#4393c3', '-',  'mART Full 360 deg'),
    ]

    unique_sizes = sorted(set(lesion_mm))
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for arc, method, color, ls, label in configs:
        ys = []
        for sz in unique_sizes:
            idxs = [i for i, (a, m, l) in enumerate(zip(arcs, methods, lesion_mm))
                    if a == arc and m == method and abs(l - sz) < 0.5]
            ys.append(cnr[idxs[0]] if idxs else np.nan)
        ax.plot(unique_sizes, ys, marker='o', linewidth=1.5, color=color,
                linestyle=ls, label=label, markersize=5)

    ax.axhline(4.0, color='k', linestyle=':', linewidth=1.2, label='Rose criterion (CNR=4)')
    ax.set_xlabel('Hemorrhage Diameter (mm)')
    ax.set_ylabel('CNR at Hemorrhage ROI')
    ax.set_title('NIH Aim 2 -- CNR vs Lesion Size\nFBP vs mART, Restricted vs Full Arc', fontsize=10)
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xticks(unique_sizes)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, 'aim2_fig02_cnr_vs_lesion.png')


# ===========================================================================
# NIH-AIM2-03: 4-panel arc comparison (full/restricted x FBP/mART)
# ===========================================================================

def fig_arc_comparison() -> None:
    paths = {
        'restricted_fbp':  DATA_DIR / 'recon_restricted_fbp.h5',
        'restricted_mart': DATA_DIR / 'recon_restricted_mart.h5',
        'full360_fbp':     DATA_DIR / 'recon_full360_fbp.h5',
        'full360_mart':    DATA_DIR / 'recon_full360_mart.h5',
    }
    for k, p in paths.items():
        if not p.exists():
            print(f'  SKIP NIH-AIM2-03: {p} not found')
            return

    vols = {}
    for k, p in paths.items():
        with h5py.File(p, 'r') as f:
            vols[k] = f['volume'][:].astype(np.float32)

    iz = HEM_VZ
    panels = [
        (vols['restricted_fbp'],  'FBP\n180 deg Restricted'),
        (vols['restricted_mart'], 'mART\n180 deg Restricted'),
        (vols['full360_fbp'],     'FBP\nFull 360 deg'),
        (vols['full360_mart'],    'mART\nFull 360 deg'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()
    for ax, (vol, title) in zip(axes_flat, panels):
        sl = _axial_slice(vol, iz)
        im = ax.imshow(sl, cmap='gray', vmin=VMIN, vmax=VMAX, origin='upper', aspect='equal')
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        hem_lx = HEM_VX - max(0, HEM_VX - 60)
        hem_ly = HEM_VY - max(0, HEM_VY - 60)
        ax.plot(hem_lx, hem_ly, 'r+', markersize=8, markeredgewidth=1.5)

    cbar = fig.colorbar(im, ax=axes_flat, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Attenuation (mm^-1)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('NIH Aim 2 -- Arc Comparison\nFBP vs mART: Restricted 180 deg vs Full 360 deg',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, 'aim2_fig03_arc_comparison.png')


# ===========================================================================
# NIH-AIM2-04: Image quality CSV table
# ===========================================================================

def fig_image_quality_table() -> None:
    recon_files = {
        'Restricted FBP':  DATA_DIR / 'recon_restricted_fbp.h5',
        'Restricted mART': DATA_DIR / 'recon_restricted_mart.h5',
        'Restricted SART': DATA_DIR / 'recon_restricted_sart.h5',
        'Full 360 FBP':    DATA_DIR / 'recon_full360_fbp.h5',
        'Full 360 mART':   DATA_DIR / 'recon_full360_mart.h5',
        'Full 360 SART':   DATA_DIR / 'recon_full360_sart.h5',
    }

    rows = []
    for label, path in recon_files.items():
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            ssim       = float(f.attrs.get('ssim', np.nan))
            psnr       = float(f.attrs.get('psnr', np.nan))
            cnr        = float(f.attrs.get('cnr_lesion', np.nan))
            wall_time  = float(f.attrs.get('wall_time_s', np.nan))
            method     = f.attrs.get('method', '?')
        rows.append({
            'Configuration': label,
            'Method': method,
            'SSIM': f'{ssim:.4f}',
            'PSNR (dB)': f'{psnr:.2f}',
            'CNR@5mm': f'{cnr:.3f}',
            'Wall Time (s)': f'{wall_time:.1f}',
        })

    if not rows:
        print('  SKIP NIH-AIM2-04: no reconstruction files found')
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIG_DIR / 'aim2_fig04_image_quality.csv'
    import csv
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved: {csv_path}')


# ===========================================================================
# NIH-AIM2-05: Simulated ROC curve
# ===========================================================================

def fig_simulated_roc() -> None:
    path = DATA_DIR / 'roc_results.h5'
    if not path.exists():
        print(f'  SKIP NIH-AIM2-05: {path} not found (ROC still running?)')
        return

    with h5py.File(path, 'r') as f:
        fpr         = f['fpr'][:]
        tpr         = f['tpr'][:]
        auc         = float(f.attrs['auc'])
        cnr_les     = f['cnr_lesion'][:]
        cnr_noles   = f['cnr_nolesion'][:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'SDSG+mART  AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random (AUC=0.5)')
    ax.axvline(0.1, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Simulated ROC Curve\n(Computational Estimate, 10+10 trials, 5mm Hemorrhage)',
                 fontsize=9)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)

    # Shade AUC
    ax.fill_between(fpr, tpr, alpha=0.15, color='blue')

    # Right: CNR distributions
    ax2 = axes[1]
    ax2.hist(cnr_les,   bins=8, alpha=0.7, color='#2166ac', label=f'Lesion (n={len(cnr_les)})')
    ax2.hist(cnr_noles, bins=8, alpha=0.7, color='#d6604d', label=f'No-lesion (n={len(cnr_noles)})')
    ax2.axvline(np.mean(cnr_les),   color='#2166ac', linestyle='--', linewidth=1.5,
                label=f'Mean les={np.mean(cnr_les):.2f}')
    ax2.axvline(np.mean(cnr_noles), color='#d6604d', linestyle='--', linewidth=1.5,
                label=f'Mean no-les={np.mean(cnr_noles):.2f}')
    ax2.set_xlabel('CNR at Hemorrhage ROI')
    ax2.set_ylabel('Count')
    ax2.set_title('Decision Score Distributions\nmART CNR at Hemorrhage Location', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle('NIH Aim 2 -- Simulated Detection Performance', fontsize=11, y=1.01)
    fig.tight_layout()
    _save(fig, 'aim2_fig05_simulated_roc.png')


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print('\n' + '=' * 60)
    print('NIH Figure Generation')
    print('=' * 60)

    print('\n[1] NIH-AIM1-03: Constellation optimization bar chart...')
    fig_constellation_optimization()

    print('\n[2] NIH-AIM1-04: mART convergence curves...')
    fig_mart_convergence()

    print('\n[3] NIH-AIM2-01: Hemorrhage detection panels...')
    fig_hemorrhage_detection()

    print('\n[4] NIH-AIM2-02: CNR vs lesion size...')
    fig_cnr_vs_lesion()

    print('\n[5] NIH-AIM2-03: Arc comparison panels...')
    fig_arc_comparison()

    print('\n[6] NIH-AIM2-04: Image quality table...')
    fig_image_quality_table()

    print('\n[7] NIH-AIM2-05: Simulated ROC curve...')
    fig_simulated_roc()

    # List generated files
    print('\n=== Generated NIH Figures ===')
    for f in sorted(FIG_DIR.glob('*')):
        size_kb = f.stat().st_size / 1000
        print(f'  {f.name:50s}  {size_kb:.1f} KB')


if __name__ == '__main__':
    main()
