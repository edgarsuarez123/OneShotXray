"""
phantom/nih_phantom.py — NIH cranial phantom for bedside hemorrhage CT simulation.

Grid: NX=200 x NY=160 x NZ=140 voxels (X x Y x Z), 1.0 mm/voxel isotropic
Physical space: X ∈ [-100, 100], Y ∈ [-80, 80], Z ∈ [-70, 70] mm
  X = left/right,  Y = inferior/superior,  Z = posterior/anterior

Attenuation coefficients from NIST XCOM at 70 keV.
"""

import numpy as np
from itertools import combinations

# Grid constants
NX, NY, NZ  = 200, 160, 140
VOXEL_SIZE  = 1.0   # mm/voxel

# 70 keV attenuation coefficients (mm^-1)
MU_BONE   = 0.048   # cortical bone
MU_BRAIN  = 0.021   # gray+white matter (~40 HU)
MU_BLOOD  = 0.023   # fresh hemorrhage (~80 HU)
MU_EDEMA  = 0.019   # edematous tissue contralateral (~20 HU)
MU_BASO4  = 0.310   # BaSO4 fiducial markers (high contrast)

# Skull geometry (mm) — prolate ellipsoid, semi-axes in (X, Y, Z)
SKULL_OUTER = np.array([90.0, 70.0, 65.0])
SKULL_INNER = SKULL_OUTER - 7.0   # 7 mm cortical shell

# Hemorrhage (primary lesion)
HEMORRHAGE_CENTER = np.array([58.0, 20.0, 0.0])   # mm from phantom center
DEFAULT_LESION_MM = 5.0                             # sphere diameter

# Edema — contralateral to hemorrhage
EDEMA_CENTER    = np.array([-58.0, 20.0, 0.0])
EDEMA_RADIUS_MM = 15.0

# Fiducial marker radius
MARKER_RADIUS_MM = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_nih_phantom(lesion_mm: float = DEFAULT_LESION_MM) -> tuple:
    """
    Build NIH cranial phantom volume and fiducial marker positions.

    Parameters
    ----------
    lesion_mm : float
        Hemorrhage sphere diameter in mm.  Pass 0 for the no-lesion control.

    Returns
    -------
    volume    : (NX, NY, NZ) float32 — attenuation coefficients (mm^-1)
    marker_3d : (8, 3) float64 — marker centers in mm
    """
    vol = np.zeros((NX, NY, NZ), dtype=np.float32)

    X, Y, Z = _coord_arrays()

    # Skull shell (outer - inner)
    outer = _ellipsoid(X, Y, Z, SKULL_OUTER)
    inner = _ellipsoid(X, Y, Z, SKULL_INNER)
    vol[outer & ~inner] = MU_BONE
    vol[inner]          = MU_BRAIN

    # Edema contralateral (overwrites brain)
    edema = _sphere(X, Y, Z, EDEMA_CENTER, EDEMA_RADIUS_MM)
    vol[inner & edema] = MU_EDEMA

    # Hemorrhage (skipped for no-lesion phantom)
    if lesion_mm > 0.0:
        hem = _sphere(X, Y, Z, HEMORRHAGE_CENTER, lesion_mm / 2.0)
        vol[inner & hem] = MU_BLOOD

    # Fiducial markers — overwrite whatever tissue is underneath
    marker_3d = _build_markers()
    for pos in marker_3d:
        m = _sphere(X, Y, Z, pos, MARKER_RADIUS_MM)
        vol[m] = MU_BASO4

    return vol, marker_3d


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _coord_arrays():
    """(X, Y, Z) coordinate arrays in mm, voxel-centered."""
    cx, cy, cz = NX / 2.0, NY / 2.0, NZ / 2.0
    xi = (np.arange(NX) - cx + 0.5) * VOXEL_SIZE
    yi = (np.arange(NY) - cy + 0.5) * VOXEL_SIZE
    zi = (np.arange(NZ) - cz + 0.5) * VOXEL_SIZE
    return np.meshgrid(xi, yi, zi, indexing='ij')   # each (NX, NY, NZ)


def _ellipsoid(X, Y, Z, semi_axes):
    ax, ay, az = semi_axes
    return (X/ax)**2 + (Y/ay)**2 + (Z/az)**2 <= 1.0


def _sphere(X, Y, Z, center_mm, radius_mm):
    r2 = (X - center_mm[0])**2 + (Y - center_mm[1])**2 + (Z - center_mm[2])**2
    return r2 <= radius_mm**2


def _build_markers() -> np.ndarray:
    """
    8 BaSO4 markers at anatomical skull surface positions.

    Bilateral pairs have deliberate Z-offsets to ensure non-coplanarity.
    Each direction vector is scaled to lie exactly on the outer skull ellipsoid.

    Returns (8, 3) float64 — marker centers in mm.
    """
    ax, ay, az = SKULL_OUTER
    dirs = np.array([
        [ 0.80, -0.50, -0.40],   # R mastoid (posterior-inferior)
        [-0.80, -0.50,  0.10],   # L mastoid (different Z → non-coplanar)
        [ 1.00,  0.10,  0.30],   # R temporal
        [-1.00,  0.10, -0.20],   # L temporal
        [ 0.50,  0.80,  0.20],   # R parietal
        [-0.50,  0.80, -0.15],   # L parietal
        [ 0.05,  1.00,  0.20],   # vertex
        [ 0.10,  0.30,  0.95],   # forehead (anterior)
    ], dtype=np.float64)

    markers = np.zeros((8, 3), dtype=np.float64)
    for i, d in enumerate(dirs):
        dx, dy, dz = d
        t = 1.0 / np.sqrt((dx/ax)**2 + (dy/ay)**2 + (dz/az)**2)
        markers[i] = t * d

    # Verify non-coplanarity: 4 points are non-coplanar iff det of 3×3 difference matrix ≠ 0
    min_det = min(
        abs(np.linalg.det((markers[list(idx)[1:]] - markers[list(idx)[0]]).T))
        for idx in combinations(range(8), 4)
    )
    assert min_det > 0.01, f"Markers near-coplanar: min_det={min_det:.5f}"

    return markers


# ---------------------------------------------------------------------------
# Standalone builder — writes all 5 variants to data/nih/
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    from pathlib import Path
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    out_dir = ROOT / 'data' / 'nih'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = ROOT / 'figures' / 'nih'
    fig_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ('lesion_5mm',  5.0),
        ('lesion_3mm',  3.0),
        ('lesion_8mm',  8.0),
        ('lesion_12mm', 12.0),
        ('nolesion',    0.0),
    ]

    for tag, d_mm in variants:
        print(f'Building phantom_{tag}.h5  (lesion={d_mm}mm)...')
        vol, markers = build_nih_phantom(d_mm)
        path = out_dir / f'phantom_{tag}.h5'
        with h5py.File(path, 'w') as f:
            f.create_dataset('volume',           data=vol,     compression='gzip')
            f.create_dataset('marker_positions', data=markers)
            f.attrs['voxel_size_mm'] = VOXEL_SIZE
            f.attrs['grid_shape']    = [NX, NY, NZ]
            f.attrs['lesion_mm']     = d_mm
            f.attrs['hemorrhage_center_mm'] = HEMORRHAGE_CENTER.tolist()
        print(f'  Saved {path}  shape={vol.shape}  mu range=[{vol.min():.4f}, {vol.max():.4f}]')

    # Inspection figure — primary (5mm lesion) phantom
    print('\nGenerating phantom inspection figure...')
    vol, markers = build_nih_phantom(5.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cx, cy, cz = NX // 2, NY // 2, NZ // 2
    # Use slice through hemorrhage
    hx = int(round(HEMORRHAGE_CENTER[0] / VOXEL_SIZE + NX / 2))
    hy = int(round(HEMORRHAGE_CENTER[1] / VOXEL_SIZE + NY / 2))

    slices = [vol[hx, :, :], vol[:, hy, :], vol[:, :, cz]]
    titles = [f'YZ plane (X={hx}px, through hemorrhage)',
              f'XZ plane (Y={hy}px, through hemorrhage)',
              f'XY plane (Z={cz}px, midline)']
    for ax, sl, title in zip(axes, slices, titles):
        im = ax.imshow(sl.T, origin='lower', cmap='gray',
                       vmin=0, vmax=MU_BONE * 1.05)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, label='μ (mm⁻¹)')
    fig.suptitle('NIH Cranial Phantom — 5mm Hemorrhage  (70 keV)', fontsize=11)
    fig.tight_layout()
    out = fig_dir / 'phantom_inspection.png'
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f'  Saved {out}')
    print('\nAll phantoms built successfully.')
