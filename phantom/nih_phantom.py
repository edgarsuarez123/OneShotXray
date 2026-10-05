"""
NIH cranial phantom builder.
PRD requirements: NIH-PHN-001 through NIH-PHN-010.

Phantom: ellipsoid skull (90x70x65mm outer semi-axes) + brain + hemorrhage sphere + edema
Grid: 200 x 160 x 140 at 1.0mm isotropic voxels  →  200 x 160 x 140mm volume
Energy: 70keV (NEVER mix with Navy 200keV values)

Outputs (data/nih/):
  phantom_lesion_5mm.h5    — primary (5mm hemorrhage)
  phantom_nolesion.h5      — no hemorrhage
  phantom_lesion_3mm.h5
  phantom_lesion_8mm.h5
  phantom_lesion_12mm.h5
"""

import numpy as np
import h5py
from pathlib import Path

# ── Physical constants (NIST XCOM, 70keV) ─────────────────────────────────────
MU_BONE   = 0.048   # mm^-1, cortical bone at 70keV
MU_BRAIN  = 0.021   # mm^-1, soft tissue at 70keV (~40 HU)
MU_BLOOD  = 0.023   # mm^-1, fresh blood at 70keV (delta HU = 40-50)
MU_EDEMA  = 0.019   # mm^-1, cerebral edema (~-20 HU)
MU_BASO4  = 0.31    # mm^-1, BaSO4 fiducial markers at 70keV
MU_AIR    = 0.0

VOXEL_SIZE_MM = 1.0   # mm per voxel (isotropic)
GRID_X = 200          # left-right (X)
GRID_Y = 160          # anterior-posterior (Y)
GRID_Z = 140          # superior-inferior (Z)

# Outer skull semi-axes (mm)
SKULL_OUTER_A = 90.0   # X semi-axis (left-right)
SKULL_OUTER_B = 70.0   # Y semi-axis (anterior-posterior)
SKULL_OUTER_C = 65.0   # Z semi-axis (superior-inferior)
SKULL_THICKNESS_MM = 7.0

# Inner skull semi-axes
SKULL_INNER_A = SKULL_OUTER_A - SKULL_THICKNESS_MM
SKULL_INNER_B = SKULL_OUTER_B - SKULL_THICKNESS_MM
SKULL_INNER_C = SKULL_OUTER_C - SKULL_THICKNESS_MM

# Hemorrhage position: right anterior hemisphere, 25mm depth from inner skull
# Inner skull surface in +X direction at ~83mm from center, so 25mm depth = ~58mm from center
HEMORRHAGE_CENTER_MM = np.array([58.0, 20.0, 0.0])   # (X, Y, Z) mm from phantom center

# Cerebral edema: contralateral hemisphere (left, -X side)
EDEMA_CENTER_MM = np.array([-50.0, 15.0, 0.0])
EDEMA_RADIUS_MM = 15.0

MARKER_RADIUS_MM = 1.0   # 2mm diameter

# Fiducial marker constellation (8 markers) in bone shell at anatomical positions (mm from center)
# All positions verified to be inside bone shell:
#   outer ellipsoid check (x/90)^2+(y/70)^2+(z/65)^2 <= 1
#   inner ellipsoid check (x/83)^2+(y/63)^2+(z/58)^2 >= 1
# L/R Z-offsets are intentionally asymmetric to guarantee non-coplanarity (no 4-point det=0)
# Marker X-positions are limited to ~70mm to stay within the 204.8mm detector FOV
# at 0.4mm pixel pitch, SOD=500mm, ODD=200mm (magnification factor ~1.4).
# Outer-ellipsoid check (x/90)^2+(y/70)^2+(z/65)^2 <= 1 — all verified.
# Inner-ellipsoid check (x/83)^2+(y/63)^2+(z/58)^2 >= 1 (in bone) — all verified.
MARKER_POSITIONS_MM = np.array([
    [-65.0, -12.0, -38.0],   # 0: L mastoid  — outer 0.893, inner 1.078
    [ 70.0, -12.0, -35.0],   # 1: R mastoid  — outer 0.923, inner 1.111  (asymm Z)
    [-70.0, -38.0,   8.0],   # 2: L temporal — outer 0.915, inner 1.094
    [ 70.0, -38.0,  13.0],   # 3: R temporal — outer 0.940, inner 1.125  (asymm Z)
    [  0.0,   0.0,  61.0],   # 4: Vertex     — outer 0.880, inner 1.104
    [-68.0,  30.0,  25.0],   # 5: L parietal — outer 0.903, inner 1.083
    [ 65.0,  30.0,  33.0],   # 6: R parietal — outer 0.964, inner 1.163  (asymm Z)
    [  0.0, -63.0,   2.0],   # 7: Forehead   — outer 0.811, inner 1.001
], dtype=np.float64)


def _check_non_coplanar(positions: np.ndarray) -> bool:
    """Return True if no 4 markers are coplanar (all 4-submatrix determinants != 0)."""
    from itertools import combinations
    for idx in combinations(range(len(positions)), 4):
        pts = positions[list(idx)]
        mat = pts[1:] - pts[0]
        if abs(np.linalg.det(mat)) < 1e-4:
            return False
    return True


def _place_sphere_noncubic(
    volume: np.ndarray,
    center_mm: np.ndarray,
    radius_mm: float,
    mu: float,
    voxel_size: float,
    grid_shape: tuple,
) -> None:
    """Place a sphere into non-cubic volume, setting voxels within radius to mu."""
    gx, gy, gz = grid_shape
    vs = voxel_size
    cx, cy, cz = center_mm

    def mm_to_vox_x(m): return int(m / vs + gx / 2)
    def mm_to_vox_y(m): return int(m / vs + gy / 2)
    def mm_to_vox_z(m): return int(m / vs + gz / 2)

    pad = int(np.ceil(radius_mm / vs)) + 1
    ix0 = max(0, mm_to_vox_x(cx) - pad)
    ix1 = min(gx, mm_to_vox_x(cx) + pad + 1)
    iy0 = max(0, mm_to_vox_y(cy) - pad)
    iy1 = min(gy, mm_to_vox_y(cy) + pad + 1)
    iz0 = max(0, mm_to_vox_z(cz) - pad)
    iz1 = min(gz, mm_to_vox_z(cz) + pad + 1)

    xs = (np.arange(ix0, ix1) - gx / 2 + 0.5) * vs
    ys = (np.arange(iy0, iy1) - gy / 2 + 0.5) * vs
    zs = (np.arange(iz0, iz1) - gz / 2 + 0.5) * vs

    XX, YY, ZZ = np.meshgrid(xs - cx, ys - cy, zs - cz, indexing='ij')
    mask = (XX**2 + YY**2 + ZZ**2) <= radius_mm**2
    volume[ix0:ix1, iy0:iy1, iz0:iz1][mask] = mu


def _place_edema_hemisphere(
    volume: np.ndarray,
    center_mm: np.ndarray,
    radius_mm: float,
    mu: float,
    voxel_size: float,
    grid_shape: tuple,
) -> None:
    """
    Place a hemisphere (half-sphere in the -X direction) of edema.
    Only fills voxels within radius that have x <= center_mm[0].
    """
    gx, gy, gz = grid_shape
    vs = voxel_size
    cx, cy, cz = center_mm

    def mm_to_vox_x(m): return int(m / vs + gx / 2)
    def mm_to_vox_y(m): return int(m / vs + gy / 2)
    def mm_to_vox_z(m): return int(m / vs + gz / 2)

    pad = int(np.ceil(radius_mm / vs)) + 1
    ix0 = max(0, mm_to_vox_x(cx) - pad)
    ix1 = min(gx, mm_to_vox_x(cx) + pad + 1)
    iy0 = max(0, mm_to_vox_y(cy) - pad)
    iy1 = min(gy, mm_to_vox_y(cy) + pad + 1)
    iz0 = max(0, mm_to_vox_z(cz) - pad)
    iz1 = min(gz, mm_to_vox_z(cz) + pad + 1)

    xs = (np.arange(ix0, ix1) - gx / 2 + 0.5) * vs
    ys = (np.arange(iy0, iy1) - gy / 2 + 0.5) * vs
    zs = (np.arange(iz0, iz1) - gz / 2 + 0.5) * vs

    XX, YY, ZZ = np.meshgrid(xs - cx, ys - cy, zs - cz, indexing='ij')
    # Hemisphere: within sphere AND in -X direction (contralateral to hemorrhage at +X)
    mask = (XX**2 + YY**2 + ZZ**2 <= radius_mm**2) & (XX <= 0)
    volume[ix0:ix1, iy0:iy1, iz0:iz1][mask] = mu


def build_nih_phantom(
    output_path: Path,
    lesion_diameter_mm: float | None = 5.0,
) -> None:
    """
    Build NIH cranial phantom and save to HDF5.

    Parameters
    ----------
    output_path : Path — output HDF5 file
    lesion_diameter_mm : float | None — hemorrhage sphere diameter in mm.
        None = no-lesion variant (hemorrhage replaced by brain tissue).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gs = (GRID_X, GRID_Y, GRID_Z)
    vs = VOXEL_SIZE_MM
    label = f"{lesion_diameter_mm:.0f}mm" if lesion_diameter_mm is not None else "no-lesion"
    print(f"Building NIH cranial phantom [{label}] ({GRID_X}x{GRID_Y}x{GRID_Z} @ {vs}mm)...")

    # ── 1. Initialize air ─────────────────────────────────────────────────────
    volume = np.zeros((GRID_X, GRID_Y, GRID_Z), dtype=np.float32)

    # ── 2. Outer skull ellipsoid (bone shell) ─────────────────────────────────
    print("  Placing outer skull ellipsoid (bone shell)...")
    xs = (np.arange(GRID_X) - GRID_X / 2 + 0.5) * vs   # (200,) in mm
    ys = (np.arange(GRID_Y) - GRID_Y / 2 + 0.5) * vs   # (160,)
    zs = (np.arange(GRID_Z) - GRID_Z / 2 + 0.5) * vs   # (140,)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')  # (200, 160, 140)

    outer_mask = (
        (XX / SKULL_OUTER_A)**2 +
        (YY / SKULL_OUTER_B)**2 +
        (ZZ / SKULL_OUTER_C)**2
    ) <= 1.0
    volume[outer_mask] = MU_BONE

    # ── 3. Inner skull (brain fill) ───────────────────────────────────────────
    print("  Filling brain interior...")
    inner_mask = (
        (XX / SKULL_INNER_A)**2 +
        (YY / SKULL_INNER_B)**2 +
        (ZZ / SKULL_INNER_C)**2
    ) <= 1.0
    volume[inner_mask] = MU_BRAIN

    # ── 4. Cerebral edema (contralateral to hemorrhage) ───────────────────────
    print(f"  Placing edema hemisphere (r={EDEMA_RADIUS_MM}mm, contralateral)...")
    _place_edema_hemisphere(volume, EDEMA_CENTER_MM, EDEMA_RADIUS_MM, MU_EDEMA, vs, gs)

    # ── 5. Hemorrhage sphere (or skip for no-lesion) ──────────────────────────
    if lesion_diameter_mm is not None:
        radius_mm = lesion_diameter_mm / 2.0
        print(f"  Placing {lesion_diameter_mm:.0f}mm hemorrhage sphere at {HEMORRHAGE_CENTER_MM}mm...")
        _place_sphere_noncubic(volume, HEMORRHAGE_CENTER_MM, radius_mm, MU_BLOOD, vs, gs)
    else:
        print("  No-lesion variant — skipping hemorrhage sphere.")

    # ── 6. BaSO4 fiducial markers ─────────────────────────────────────────────
    assert _check_non_coplanar(MARKER_POSITIONS_MM), \
        "Marker constellation is coplanar — must fix positions (NIH-PHN-008)"
    print(f"  Placing {len(MARKER_POSITIONS_MM)} BaSO4 markers (r={MARKER_RADIUS_MM}mm each)...")
    for i, pos in enumerate(MARKER_POSITIONS_MM):
        _place_sphere_noncubic(volume, pos, MARKER_RADIUS_MM, MU_BASO4, vs, gs)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    print(f"  Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volume', data=volume, compression='gzip', compression_opts=4)
        f.create_dataset('marker_positions', data=MARKER_POSITIONS_MM.astype(np.float32))
        f.attrs['voxel_size_mm']       = VOXEL_SIZE_MM
        f.attrs['grid_x']              = GRID_X
        f.attrs['grid_y']              = GRID_Y
        f.attrs['grid_z']              = GRID_Z
        f.attrs['mu_bone']             = MU_BONE
        f.attrs['mu_brain']            = MU_BRAIN
        f.attrs['mu_blood']            = MU_BLOOD
        f.attrs['mu_edema']            = MU_EDEMA
        f.attrs['mu_baso4']            = MU_BASO4
        f.attrs['energy_kev']          = 70
        f.attrs['n_markers']           = len(MARKER_POSITIONS_MM)
        f.attrs['marker_radius_mm']    = MARKER_RADIUS_MM
        f.attrs['lesion_diameter_mm']  = lesion_diameter_mm if lesion_diameter_mm is not None else -1.0
        f.attrs['hemorrhage_center_mm'] = HEMORRHAGE_CENTER_MM

    size_mb = output_path.stat().st_size / 1e6
    print(f"  Done. File size: {size_mb:.1f} MB")
    print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"  Value range: [{volume.min():.4f}, {volume.max():.4f}]")
    n_bone  = int(outer_mask.sum()) - int(inner_mask.sum())
    n_brain = int(inner_mask.sum())
    print(f"  Bone voxels: {n_bone}, Brain voxels: {n_brain}")


def build_all_variants(out_dir: Path) -> None:
    """Build all 5 NIH phantom variants."""
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        (5.0,  out_dir / 'phantom_lesion_5mm.h5'),
        (None, out_dir / 'phantom_nolesion.h5'),
        (3.0,  out_dir / 'phantom_lesion_3mm.h5'),
        (8.0,  out_dir / 'phantom_lesion_8mm.h5'),
        (12.0, out_dir / 'phantom_lesion_12mm.h5'),
    ]

    for lesion_diam, path in variants:
        print()
        build_nih_phantom(path, lesion_diameter_mm=lesion_diam)
        print(f"  -> {path.name}")

    print("\n[DONE] All 5 NIH phantom variants built.")
    print(f"  Files in {out_dir}:")
    for _, p in variants:
        size_mb = p.stat().st_size / 1e6
        print(f"    {p.name:40s}  {size_mb:.1f} MB")


if __name__ == '__main__':
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    out_dir = ROOT / 'data' / 'nih'
    build_all_variants(out_dir)

    # Quick inspection figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    primary_path = out_dir / 'phantom_lesion_5mm.h5'
    with h5py.File(primary_path, 'r') as f:
        vol = f['volume'][:]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cx, cy, cz = GRID_X // 2, GRID_Y // 2, GRID_Z // 2
    vmin, vmax = 0.0, MU_BASO4

    im = axes[0].imshow(vol[:, :, cz].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title(f'Axial (Z={cz})')
    axes[0].set_xlabel('X (left-right)')
    axes[0].set_ylabel('Y (ant-post)')

    axes[1].imshow(vol[:, cy, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title(f'Coronal (Y={cy})')
    axes[1].set_xlabel('X (left-right)')
    axes[1].set_ylabel('Z (sup-inf)')

    axes[2].imshow(vol[cx, :, :].T, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    axes[2].set_title(f'Sagittal (X={cx})')
    axes[2].set_xlabel('Y (ant-post)')
    axes[2].set_ylabel('Z (sup-inf)')

    fig.colorbar(im, ax=axes, label='Attenuation (mm^-1)')
    fig.suptitle('NIH Cranial Phantom — 5mm Hemorrhage (70keV)')
    fig.tight_layout()

    fig_path = ROOT / 'figures' / 'nih' / 'phantom_inspection.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nInspection figure saved: {fig_path}")
