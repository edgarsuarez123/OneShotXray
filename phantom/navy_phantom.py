"""
Navy steel phantom builder.
PRD requirements: NV-PHN-001 through NV-PHN-009.

Phantom: 25mm x 25mm x 25mm steel block
Grid: 250 x 250 x 250 at 0.1mm isotropic voxels
Output: data/navy/phantom.h5
"""

import itertools
import numpy as np
import h5py
from pathlib import Path


# ── Physical constants (NIST XCOM, 200keV) ────────────────────────────────────
MU_STEEL  = 0.1149   # mm^-1, iron at 200keV
MU_AIR    = 0.0      # mm^-1, air void (cracks and porosity)
MU_LEAD   = 1.133    # mm^-1, lead at 200keV (fiducial markers)

VOXEL_SIZE_MM  = 0.1   # mm per voxel (isotropic)
GRID_SIZE      = 250   # voxels per side → 25mm phantom

# Crack definitions: (width_mm, depth_mm_from_front_face)
CRACKS = [
    (0.4,  12.0),   # below-spec stretch target (Phase II)
    (0.8,  12.0),   # primary ASTM E1742 target
    (1.6,  12.0),   # sanity check, clearly visible
]

# Fiducial marker constellation (8 spheres, 2mm diameter)
# Positions in phantom coordinate space (mm from phantom center = 0,0,0)
# Non-coplanar, non-symmetric — verified below
MARKER_POSITIONS_MM = np.array([
    [-9.0,  -9.0,  -9.0],
    [ 9.0,  -8.5,  -8.0],
    [-8.5,   9.0,  -7.5],
    [ 8.0,   8.5,  -9.0],
    [-9.0,  -8.0,   9.0],
    [ 9.0,  -9.0,   8.5],
    [-8.5,   8.0,   8.0],
    [ 8.5,   9.0,   9.0],
], dtype=np.float32)

MARKER_RADIUS_MM = 1.0   # 2mm diameter → 1mm radius

# Gas porosity voids (spherical, mu=0)
POROSITY_POSITIONS_MM = np.array([
    [-4.0,  4.0, -3.0],   # 0.5mm diameter void
    [ 5.0, -3.0,  4.0],   # 1.0mm diameter void
], dtype=np.float32)
POROSITY_RADII_MM = np.array([0.25, 0.50], dtype=np.float32)  # radius = diameter/2


def _check_non_coplanar(positions: np.ndarray) -> bool:
    """Return True if no 4 markers are coplanar (all 4-submatrix determinants != 0)."""
    from itertools import combinations
    for idx in combinations(range(len(positions)), 4):
        pts = positions[list(idx)]
        # Vectors from first point
        mat = pts[1:] - pts[0]
        if abs(np.linalg.det(mat)) < 1e-6:
            return False
    return True


def _build_crack_plane(volume: np.ndarray, crack_width_mm: float, crack_depth_mm: float) -> None:
    """
    Carve a crack plane (XZ plane, perpendicular to Y axis) into the volume.
    The crack is a planar air void running the full length of the phantom.
    Uses sub-voxel partial volume weighting (NV-PHN-006).

    crack_depth_mm: distance from the -Y face to crack center
    crack_width_mm: full width of crack in mm
    """
    n = GRID_SIZE
    vs = VOXEL_SIZE_MM

    # Crack center in voxel index space (Y axis)
    # Phantom spans [-12.5mm, +12.5mm]; voxel i corresponds to center at (i - n/2 + 0.5)*vs
    crack_center_y_mm = -12.5 + crack_depth_mm  # in phantom coords

    half_width = crack_width_mm / 2.0

    for iy in range(n):
        voxel_center_y = (iy - n / 2 + 0.5) * vs  # mm
        dist = abs(voxel_center_y - crack_center_y_mm)
        if dist < half_width + vs:
            # Fraction of voxel that lies inside the crack
            overlap = min(voxel_center_y + vs / 2, crack_center_y_mm + half_width) - \
                      max(voxel_center_y - vs / 2, crack_center_y_mm - half_width)
            fraction = max(0.0, overlap / vs)
            if fraction > 0:
                volume[:, iy, :] = MU_STEEL * (1.0 - fraction)


def _place_sphere(volume: np.ndarray, center_mm: np.ndarray, radius_mm: float, mu: float) -> None:
    """Replace voxels within radius_mm of center_mm with mu."""
    n = GRID_SIZE
    vs = VOXEL_SIZE_MM
    cx, cy, cz = center_mm

    # Bounding box in voxel indices
    def mm_to_vox(m):
        return int((m / vs) + n / 2)

    pad = int(np.ceil(radius_mm / vs)) + 1
    ix0 = max(0, mm_to_vox(cx) - pad)
    ix1 = min(n, mm_to_vox(cx) + pad + 1)
    iy0 = max(0, mm_to_vox(cy) - pad)
    iy1 = min(n, mm_to_vox(cy) + pad + 1)
    iz0 = max(0, mm_to_vox(cz) - pad)
    iz1 = min(n, mm_to_vox(cz) + pad + 1)

    xs = (np.arange(ix0, ix1) - n / 2 + 0.5) * vs
    ys = (np.arange(iy0, iy1) - n / 2 + 0.5) * vs
    zs = (np.arange(iz0, iz1) - n / 2 + 0.5) * vs

    XX, YY, ZZ = np.meshgrid(xs - cx, ys - cy, zs - cz, indexing='ij')
    mask = (XX**2 + YY**2 + ZZ**2) <= radius_mm**2
    volume[ix0:ix1, iy0:iy1, iz0:iz1][mask] = mu


def build_navy_phantom(output_path: Path) -> None:
    """Build the Navy steel phantom and save to HDF5."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building Navy steel phantom ({GRID_SIZE}^3 @ {VOXEL_SIZE_MM}mm)...")

    # ── 1. Steel block ────────────────────────────────────────────────────────
    volume = np.full((GRID_SIZE, GRID_SIZE, GRID_SIZE), MU_STEEL, dtype=np.float32)

    # ── 2. Crack planes (partial volume) ─────────────────────────────────────
    for width_mm, depth_mm in CRACKS:
        print(f"  Carving {width_mm}mm crack at depth {depth_mm}mm...")
        _build_crack_plane(volume, width_mm, depth_mm)

    # ── 3. Lead fiducial markers ──────────────────────────────────────────────
    assert _check_non_coplanar(MARKER_POSITIONS_MM), \
        "Marker constellation is coplanar — must fix positions (NV-PHN-007)"
    for i, pos in enumerate(MARKER_POSITIONS_MM):
        print(f"  Placing marker {i+1}/{len(MARKER_POSITIONS_MM)} at {pos}mm...")
        _place_sphere(volume, pos, MARKER_RADIUS_MM, MU_LEAD)

    # ── 4. Gas porosity voids ─────────────────────────────────────────────────
    for i, (pos, rad) in enumerate(zip(POROSITY_POSITIONS_MM, POROSITY_RADII_MM)):
        diam_mm = rad * 2
        print(f"  Placing {diam_mm:.1f}mm porosity void {i+1}...")
        _place_sphere(volume, pos, float(rad), MU_AIR)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    crack_widths  = np.array([c[0] for c in CRACKS], dtype=np.float32)
    crack_depths  = np.array([c[1] for c in CRACKS], dtype=np.float32)

    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volume',           data=volume,                 compression='gzip', compression_opts=4)
        f.create_dataset('marker_positions', data=MARKER_POSITIONS_MM)
        f.create_dataset('crack_widths_mm',  data=crack_widths)
        f.create_dataset('crack_depths_mm',  data=crack_depths)
        f.attrs['voxel_size_mm']   = VOXEL_SIZE_MM
        f.attrs['grid_size']       = GRID_SIZE
        f.attrs['mu_steel']        = MU_STEEL
        f.attrs['mu_lead']         = MU_LEAD
        f.attrs['energy_kev']      = 200
        f.attrs['n_markers']       = len(MARKER_POSITIONS_MM)
        f.attrs['marker_radius_mm'] = MARKER_RADIUS_MM

    size_mb = output_path.stat().st_size / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
    print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    print(f"Value range: [{volume.min():.4f}, {volume.max():.4f}]")


if __name__ == '__main__':
    output = Path(__file__).parent.parent / 'data' / 'navy' / 'phantom.h5'
    build_navy_phantom(output)
