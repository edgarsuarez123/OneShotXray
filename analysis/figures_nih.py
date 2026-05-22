"""
analysis/figures_nih.py -Generate all NIH proposal figures (300 DPI).

Figures produced:
  aim1_fig01_constellation_optimization.png  -solver residual vs marker count × arc
  aim1_fig02_mart_convergence.png            -mART convergence curves (λ sweep)
  aim2_fig01_hemorrhage_detection.png        -GT | FBP | mART slices at hemorrhage
  aim2_fig02_cnr_vs_lesion.png              -CNR vs lesion diameter
  aim2_fig03_arc_comparison.png             -restricted vs full360 × method
  aim2_fig04_image_quality.csv             -full metrics table
  aim2_fig05_simulated_roc.png             -ROC curve (n=200) with CI band + density inset
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / 'figures' / 'nih'
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# ── Helpers ─────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


# ── Aim 1: Constellation grid ────────────────────────────────────────────────

def fig_aim1_constellation():
    path = ROOT / 'data' / 'nih' / 'aim1_constellation_grid.h5'
    if not path.exists():
        print(f'  SKIP aim1_fig01 -{path} not found')
        return

    with h5py.File(path, 'r') as f:
        arcs    = list(f.keys())
        markers = [4, 6, 8]
        data = {}
        for arc in arcs:
            data[arc] = [f[arc].attrs.get(f'markers_{m}_mean_residual_px', np.nan)
                         for m in markers]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(markers))
    w = 0.35
    bars_r = ax.bar(x - w/2, data.get('restricted', [np.nan]*3), w,
                    label='Restricted 180°', color='steelblue')
    bars_f = ax.bar(x + w/2, data.get('full360', [np.nan]*3), w,
                    label='Full 360°', color='darkorange')
    ax.axhline(0.3, color='red', ls='--', lw=1, label='Gate 0.3 px')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m} markers' for m in markers])
    ax.set_ylabel('Mean solver residual (px)')
    ax.set_title('Aim 1 -SDSG Residual vs Marker Count × Arc')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.35)
    fig.tight_layout()
    _save(fig, 'aim1_fig01_constellation_optimization.png')


# ── Aim 1: mART convergence ──────────────────────────────────────────────────

def fig_aim1_mart_convergence():
    path = ROOT / 'data' / 'nih' / 'aim1_mart_sweep.h5'
    if not path.exists():
        print(f'  SKIP aim1_fig02 -{path} not found')
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    iter_vals = [25, 50, 100]
    lam_vals  = [0.5, 1.0, 2.0]
    colors    = ['#1f77b4', '#ff7f0e', '#2ca02c']

    with h5py.File(path, 'r') as f:
        for ax, n_iter in zip(axes, iter_vals):
            for lam, col in zip(lam_vals, colors):
                key = f'iter_{n_iter}/lam_{lam}'
                if key not in f:
                    continue
                conv = f[key]['convergence'][:]
                cnr  = f[key].attrs.get('cnr', np.nan)
                ax.semilogy(np.arange(1, len(conv)+1), conv, color=col,
                            label=f'λ={lam}  CNR={cnr:.2f}')
            ax.set_title(f'{n_iter} iterations')
            ax.set_xlabel('Iteration')
            if ax is axes[0]:
                ax.set_ylabel('||correction|| / ||x||')
            ax.legend(fontsize=8)
    fig.suptitle('Aim 1 -mART Convergence (restricted arc, 5mm lesion)', y=1.01)
    fig.tight_layout()
    _save(fig, 'aim1_fig02_mart_convergence.png')


# ── Aim 2: hemorrhage detection slices ──────────────────────────────────────

def fig_aim2_hemorrhage_detection():
    recon_path   = ROOT / 'data' / 'nih' / 'recon_restricted.h5'
    phantom_path = ROOT / 'data' / 'nih' / 'phantom_lesion_5mm.h5'
    if not recon_path.exists() or not phantom_path.exists():
        print(f'  SKIP aim2_fig01 -data missing')
        return

    from phantom.nih_phantom import HEMORRHAGE_CENTER, VOXEL_SIZE, NX, NY

    with h5py.File(phantom_path, 'r') as f:
        phantom = f['volume'][:]
    with h5py.File(recon_path, 'r') as f:
        fbp_vol  = f['fbp/volume'][:]
        mart_vol = f['mart/volume'][:]
        mart_cnr = f['mart'].attrs.get('cnr', 0)

    # Axial slice through hemorrhage center
    hy = int(round(HEMORRHAGE_CENTER[1] / VOXEL_SIZE + NY / 2))
    slices = [phantom[:, hy, :], fbp_vol[:, hy, :], mart_vol[:, hy, :]]
    titles = ['Ground truth phantom',
              'FBP (restricted arc)',
              f'mART 25-iter λ=0.5\nCNR@5mm = {mart_cnr:.2f}']

    vmax = float(phantom.max()) * 1.05
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, sl, title in zip(axes, slices, titles):
        im = ax.imshow(sl.T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='μ (mm⁻¹)')

    # Mark hemorrhage ROI
    hx_px = int(round(HEMORRHAGE_CENTER[0] / VOXEL_SIZE + phantom.shape[0] / 2))
    hz_px = int(round(HEMORRHAGE_CENTER[2] / VOXEL_SIZE + phantom.shape[2] / 2))
    for ax in axes:
        circle = plt.Circle((hx_px, hz_px), 4, color='red', fill=False, lw=1.5,
                             transform=ax.transData)
        ax.add_patch(circle)

    fig.suptitle('Aim 2 -Hemorrhage Detection: Restricted 180° Arc (5mm lesion, 70 keV)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, 'aim2_fig01_hemorrhage_detection.png')


# ── Aim 2: CNR vs lesion size ────────────────────────────────────────────────

def fig_aim2_cnr_vs_lesion():
    path = ROOT / 'data' / 'nih' / 'recon_lesion_sweep.h5'
    if not path.exists():
        print(f'  SKIP aim2_fig02 -{path} not found')
        return

    lesions = [3.0, 5.0, 8.0, 12.0]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'restricted': 'steelblue', 'full360': 'darkorange'}

    with h5py.File(path, 'r') as f:
        for arc, col in colors.items():
            if arc not in f:
                continue
            cnrs = [f[arc][f'lesion_{d}mm'].attrs.get('cnr', np.nan) for d in lesions]
            ax.plot(lesions, cnrs, 'o-', color=col, label=arc.replace('360', ' 360°'), lw=2)

    ax.axhline(4.0, color='red', ls='--', lw=1.5, label='Rose criterion (CNR=4)')
    ax.set_xlabel('Hemorrhage diameter (mm)')
    ax.set_ylabel('CNR at hemorrhage ROI')
    ax.set_title('Aim 2 -Detectability vs Lesion Size (mART, restricted vs full arc)')
    ax.legend()
    ax.set_xticks(lesions)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, 'aim2_fig02_cnr_vs_lesion.png')


# ── Aim 2: arc comparison ────────────────────────────────────────────────────

def fig_aim2_arc_comparison():
    r_path = ROOT / 'data' / 'nih' / 'recon_restricted.h5'
    f_path = ROOT / 'data' / 'nih' / 'recon_full360.h5'
    p_path = ROOT / 'data' / 'nih' / 'phantom_lesion_5mm.h5'
    if not r_path.exists() or not f_path.exists() or not p_path.exists():
        print('  SKIP aim2_fig03 -data missing')
        return

    from phantom.nih_phantom import HEMORRHAGE_CENTER, VOXEL_SIZE, NY

    with h5py.File(p_path,  'r') as f: phantom  = f['volume'][:]
    with h5py.File(r_path, 'r') as f:
        mart_r  = f['mart/volume'][:]
        fbp_r   = f['fbp/volume'][:]
        cnr_r   = f['mart'].attrs.get('cnr', 0)
    with h5py.File(f_path, 'r') as f:
        mart_f  = f['mart/volume'][:]
        fbp_f   = f['fbp/volume'][:]
        cnr_f   = f['mart'].attrs.get('cnr', 0)

    hy   = int(round(HEMORRHAGE_CENTER[1] / VOXEL_SIZE + NY / 2))
    vmax = float(phantom.max()) * 1.05
    panels = [
        (fbp_r,  f'FBP\nRestricted'),
        (mart_r, f'mART\nRestricted\nCNR={cnr_r:.2f}'),
        (fbp_f,  f'FBP\nFull 360°'),
        (mart_f, f'mART\nFull 360°\nCNR={cnr_f:.2f}'),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (vol, title) in zip(axes, panels):
        im = ax.imshow(vol[:, hy, :].T, origin='lower', cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='μ (mm⁻¹)')
    fig.suptitle('Aim 2 -Arc Comparison: Restricted 180° vs Full 360° (5mm lesion)',
                 fontsize=10)
    fig.tight_layout()
    _save(fig, 'aim2_fig03_arc_comparison.png')


# ── Aim 2: image quality table ───────────────────────────────────────────────

def fig_aim2_image_quality_csv():
    rows = []
    for arc in ('restricted', 'full360'):
        path = ROOT / 'data' / 'nih' / f'recon_{arc}.h5'
        if not path.exists():
            continue
        with h5py.File(path, 'r') as f:
            for method in ('fbp', 'mart', 'sart'):
                if method not in f:
                    continue
                rows.append({
                    'arc':    arc,
                    'method': method.upper(),
                    'ssim':   f'{f[method].attrs.get("ssim", float("nan")):.4f}',
                    'psnr_dB': f'{f[method].attrs.get("psnr", float("nan")):.2f}',
                    'cnr_5mm': f'{f[method].attrs.get("cnr", float("nan")):.4f}',
                })

    out = FIG_DIR / 'aim2_fig04_image_quality.csv'
    with open(out, 'w', newline='') as csvf:
        w = csv.DictWriter(csvf, fieldnames=['arc', 'method', 'ssim', 'psnr_dB', 'cnr_5mm'])
        w.writeheader()
        w.writerows(rows)
    print(f'  Saved {out}')


# ── Aim 2: ROC curve ────────────────────────────────────────────────────────

def fig_aim2_roc():
    path = ROOT / 'data' / 'nih' / 'roc_results.h5'
    if not path.exists():
        print(f'  SKIP aim2_fig05 -{path} not found')
        return

    from analysis.run_nih_roc import compute_auc, hanley_mcneil_ci

    with h5py.File(path, 'r') as f:
        if 'cnr_lesion' not in f or 'cnr_nolesion' not in f:
            print('  SKIP aim2_fig05 - ROC sweep still running (cnr_nolesion missing)')
            return
        cnr_l  = f['cnr_lesion'][:]
        cnr_nl = f['cnr_nolesion'][:]
        auc    = float(f.attrs.get('auc', 0))
        ci_lo  = float(f.attrs.get('auc_ci_lo', 0))
        ci_hi  = float(f.attrs.get('auc_ci_hi', 1))

    cnr_l_v  = cnr_l[~np.isnan(cnr_l)]
    cnr_nl_v = cnr_nl[~np.isnan(cnr_nl)]

    # Build ROC curve by sweeping threshold
    all_scores = np.concatenate([cnr_l_v, cnr_nl_v])
    thresholds = np.sort(np.unique(all_scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        tprs.append(np.mean(cnr_l_v >= t))
        fprs.append(np.mean(cnr_nl_v >= t))
    tprs.append(1.0); fprs.append(1.0)
    tprs = np.array(tprs); fprs = np.array(fprs)

    # Optimal threshold (Youden index)
    j = tprs - fprs
    opt_idx = int(np.argmax(j[1:-1])) + 1
    opt_t   = thresholds[opt_idx - 1]
    sens    = float(tprs[opt_idx])
    spec    = float(1 - fprs[opt_idx])

    fig = plt.figure(figsize=(9, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax_roc  = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1])

    # ROC with CI shading
    ax_roc.plot(fprs, tprs, 'b-', lw=2, label=f'ROC (AUC={auc:.3f})')
    ax_roc.fill_between([0, 1], [ci_lo, ci_lo], [ci_hi, ci_hi],
                         alpha=0.15, color='blue',
                         label=f'95% CI [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax_roc.plot(fprs[opt_idx], tprs[opt_idx], 'r*', ms=12,
                label=f'Optimal thr={opt_t:.3f}\nSens={sens:.0%} Spec={spec:.0%}')
    ax_roc.set_xlabel('False Positive Rate (1 − Specificity)')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)')
    ax_roc.set_title(f'Aim 2 -Simulated ROC\n'
                     f'n={len(cnr_l_v)+len(cnr_nl_v)} trials  AUC={auc:.3f}')
    ax_roc.legend(fontsize=8, loc='lower right')
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # CNR score distributions
    bins = np.linspace(min(cnr_nl_v.min(), cnr_l_v.min()) - 0.5,
                       max(cnr_nl_v.max(), cnr_l_v.max()) + 0.5, 30)
    ax_dist.hist(cnr_nl_v, bins=bins, alpha=0.6, color='orange', label='No lesion')
    ax_dist.hist(cnr_l_v,  bins=bins, alpha=0.6, color='steelblue', label='5mm lesion')
    ax_dist.axvline(opt_t, color='red', ls='--', lw=1.5, label=f'Threshold={opt_t:.3f}')
    ax_dist.set_xlabel('CNR at hemorrhage ROI')
    ax_dist.set_ylabel('Count')
    ax_dist.set_title('CNR score distributions')
    ax_dist.legend(fontsize=8)

    fig.suptitle('SDSG + mART Bedside CT -Hemorrhage Detection ROC (n=200)', fontsize=10)
    _save(fig, 'aim2_fig05_simulated_roc.png')


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating NIH figures...')
    fig_aim1_constellation()
    fig_aim1_mart_convergence()
    fig_aim2_hemorrhage_detection()
    fig_aim2_cnr_vs_lesion()
    fig_aim2_arc_comparison()
    fig_aim2_image_quality_csv()
    fig_aim2_roc()
    print('\nAll NIH figures generated.')
