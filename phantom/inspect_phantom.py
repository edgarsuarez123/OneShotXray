"""
Visual inspection of the Navy steel phantom.
Produces a 3-panel figure (axial, coronal, sagittal) saved to figures/navy/phantom_inspection.png.
"""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


PHANTOM_PATH = Path(__file__).parent.parent / 'data' / 'navy' / 'phantom.h5'
OUTPUT_PATH  = Path(__file__).parent.parent / 'figures' / 'navy' / 'phantom_inspection.png'


def inspect(phantom_path: Path = PHANTOM_PATH, output_path: Path = OUTPUT_PATH) -> None:
    with h5py.File(phantom_path, 'r') as f:
        volume          = f['volume'][:]
        marker_pos      = f['marker_positions'][:]
        crack_widths    = f['crack_widths_mm'][:]
        crack_depths    = f['crack_depths_mm'][:]
        voxel_size      = f.attrs['voxel_size_mm']
        n               = f.attrs['grid_size']

    n = volume.shape[0]
    mid = n // 2  # center slice index

    # Crack center in voxel index (Y axis)
    # Phantom Y goes from -12.5mm (iy=0) to +12.5mm (iy=249)
    # crack_depth is from -Y face, so crack_center_y_mm = -12.5 + depth
    def depth_to_iy(depth_mm):
        crack_center_y_mm = -12.5 + depth_mm
        return int((crack_center_y_mm / voxel_size) + n / 2)

    crack_iy = [depth_to_iy(d) for d in crack_depths]

    # ── Marker positions in voxel coords ─────────────────────────────────────
    def mm_to_vox(m):
        return (m / voxel_size) + n / 2

    marker_vox = mm_to_vox(marker_pos)  # shape (8, 3): columns are ix, iy, iz

    # ── Figures ───────────────────────────────────────────────────────────────
    matplotlib.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Navy Steel Phantom — Visual Inspection', fontsize=14, fontweight='bold')

    # Steel/air window — clamp to [0, mu_steel]
    vmax = 0.1149

    # Panel A: Axial slice (XY plane at mid Z)
    ax = axes[0]
    slc = volume[:, :, mid].T   # (Y, X)
    im = ax.imshow(slc, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                   extent=[-12.5, 12.5, -12.5, 12.5])
    ax.set_title(f'Axial (Z={0:.0f}mm)')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    # Mark crack lines (horizontal lines at crack Y positions)
    for w, iy in zip(crack_widths, crack_iy):
        y_mm = (iy - n / 2 + 0.5) * voxel_size
        ax.axhline(y_mm, color='red', linewidth=0.8, linestyle='--', alpha=0.8)
        ax.text(11, y_mm + 0.3, f'{w}mm', color='red', fontsize=8)
    # Mark marker positions (project onto XY)
    for mk in marker_vox:
        x_mm = (mk[0] - n/2 + 0.5) * voxel_size
        y_mm = (mk[1] - n/2 + 0.5) * voxel_size
        ax.plot(x_mm, y_mm, 'o', color='cyan', markersize=4, markeredgewidth=0.5)
    plt.colorbar(im, ax=ax, label='mu (mm^-1)', fraction=0.046)

    # Panel B: Coronal slice (XZ plane at crack depth Y index for primary 0.8mm crack)
    ax = axes[1]
    primary_iy = crack_iy[1]  # 0.8mm crack index
    slc = volume[:, primary_iy, :].T   # (Z, X)
    im = ax.imshow(slc, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                   extent=[-12.5, 12.5, -12.5, 12.5])
    ax.set_title('Coronal -- through 0.8mm crack')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.text(-11, 11, 'Porosity voids\n(dark circles)', color='yellow', fontsize=8)
    plt.colorbar(im, ax=ax, label='mu (mm^-1)', fraction=0.046)

    # Panel C: Sagittal slice (YZ plane at mid X) — shows all 3 crack planes
    ax = axes[2]
    slc = volume[mid, :, :].T   # (Z, Y)
    im = ax.imshow(slc, cmap='gray', vmin=0, vmax=vmax, origin='lower',
                   extent=[-12.5, 12.5, -12.5, 12.5])
    ax.set_title('Sagittal (X=0) -- all 3 cracks')
    ax.set_xlabel('Y (mm)')
    ax.set_ylabel('Z (mm)')
    for w, iy in zip(crack_widths, crack_iy):
        y_mm = (iy - n / 2 + 0.5) * voxel_size
        ax.axvline(y_mm, color='red', linewidth=0.8, linestyle='--', alpha=0.9)
        ax.text(y_mm + 0.2, 10, f'{w}mm', color='red', fontsize=8, rotation=90)
    plt.colorbar(im, ax=ax, label='mu (mm^-1)', fraction=0.046)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved inspection figure: {output_path}")

    # ── Console QA report ─────────────────────────────────────────────────────
    print("\n=== PHANTOM QA REPORT ===")
    print(f"Volume shape:     {volume.shape}")
    print(f"Voxel size:       {voxel_size} mm")
    print(f"Physical size:    {volume.shape[0]*voxel_size:.1f} x "
          f"{volume.shape[1]*voxel_size:.1f} x {volume.shape[2]*voxel_size:.1f} mm")
    print(f"Value range:      [{volume.min():.4f}, {volume.max():.4f}] mm^-1")
    print(f"Expected steel mu: {0.1149:.4f} mm^-1")
    print(f"Expected lead mu:  {1.133:.4f} mm^-1")
    print(f"\nCracks:")
    for w, d, iy in zip(crack_widths, crack_depths, crack_iy):
        y_mm = (iy - n/2 + 0.5) * voxel_size
        vox_width = int(w / voxel_size)
        print(f"  {w}mm crack — depth {d}mm — voxel iy~{iy} — ~{vox_width} voxels wide")
    print(f"\nMarkers: {len(marker_pos)} spheres, 2mm diameter")
    print(f"  Non-coplanarity check:", end=' ')
    from itertools import combinations
    ok = True
    for idx in combinations(range(len(marker_pos)), 4):
        pts = marker_pos[list(idx)]
        mat = pts[1:] - pts[0]
        if abs(np.linalg.det(mat)) < 1e-6:
            ok = False
            break
    print("PASS" if ok else "FAIL — fix marker positions!")
    print(f"\nPorosity voids: {len(POROSITY_RADII) if 'POROSITY_RADII' in dir() else 2} spheres")


POROSITY_RADII = np.array([0.25, 0.50])

if __name__ == '__main__':
    inspect()
