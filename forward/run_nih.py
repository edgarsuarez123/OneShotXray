"""
forward/run_nih.py — NIH forward projection pipeline (restricted + full arc).

Reads:  data/nih/phantom_lesion_5mm.h5
Writes: data/nih/sinogram_80_restricted.h5
        data/nih/sinogram_80_full360.h5

Both files share the same phantom + geometry parameters; the only difference is
whether the shots cover phi ∈ [0, π] (restricted arc) or phi ∈ [0, 2π] (full).
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forward.geometry import perturb_geometry, geometry_to_cone_vec
from forward.projector import forward_project, apply_noise_pipeline
from forward.nih_centroiding import simulate_centroids
from forward.nih_geometry import (
    generate_restricted_arc_shots,
    generate_full_arc_shots,
    SOD, ODD, SIGMA_S, SIGMA_THETA, N_SHOTS,
    DET_ROWS, DET_COLS, DET_SPACING, I0,
)
from phantom.nih_phantom import NX, NY, NZ, VOXEL_SIZE


def run_nih_forward(arc: str = 'restricted') -> None:
    """
    Parameters
    ----------
    arc : 'restricted' — 180° lateral arc (ICU bedside)
          'full360'    — full hemisphere (comparison baseline)
    """
    assert arc in ('restricted', 'full360')
    t0 = time.perf_counter()

    phantom_path = ROOT / 'data' / 'nih' / 'phantom_lesion_5mm.h5'
    out_path     = ROOT / 'data' / 'nih' / f'sinogram_80_{arc}.h5'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'NIH Forward Projection — {arc} arc')
    print('=' * 60)

    # Load phantom
    print(f'\n[1/4] Loading {phantom_path}')
    with h5py.File(phantom_path, 'r') as f:
        volume    = f['volume'][:]            # (NX, NY, NZ) float32
        marker_3d = f['marker_positions'][:]  # (8, 3) float64
    print(f'      volume shape {volume.shape}, markers {marker_3d.shape}')

    # Generate source positions
    print(f'\n[2/4] Generating {N_SHOTS} {arc} shots...')
    if arc == 'restricted':
        src_positions = generate_restricted_arc_shots(N_SHOTS, SOD)
    else:
        src_positions = generate_full_arc_shots(N_SHOTS, SOD)

    gt_9dof = perturb_geometry(src_positions, SOD, ODD, SIGMA_S, SIGMA_THETA, seed=42)
    vectors = geometry_to_cone_vec(gt_9dof, DET_SPACING, DET_ROWS, DET_COLS)

    actual_sigma_s = float(np.mean(np.linalg.norm(
        gt_9dof[:, 0:3] - src_positions, axis=1)))
    print(f'      Mean position perturbation: {actual_sigma_s:.2f} mm')

    # Forward project (clean, no noise)
    print(f'\n[3/4] Forward projection ({NX}×{NY}×{NZ} phantom, {N_SHOTS} shots)...')
    t_fp = time.perf_counter()
    sino_clean = forward_project(volume, vectors, VOXEL_SIZE, DET_ROWS, DET_COLS)
    print(f'      Done in {time.perf_counter() - t_fp:.1f}s  '
          f'shape={sino_clean.shape}  central={sino_clean[DET_ROWS//2, 0, DET_COLS//2]:.4f}')

    # Apply noise (seed=42 — primary sinogram for AIM 1/2, not ROC)
    sino_noisy = apply_noise_pipeline(
        sino_clean, I0=I0, focal_spot=0.5,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING, seed=42,
    )

    # Simulate centroids
    print('\n[4/4] Simulating CRB centroids...')
    positions, detection_mask, weights = simulate_centroids(
        gt_9dof, marker_3d, DET_SPACING, DET_ROWS, DET_COLS, seed=1042,
    )
    det_rate = detection_mask.sum() / detection_mask.size
    print(f'      Detection rate: {det_rate:.1%}  '
          f'({detection_mask.sum()}/{detection_mask.size})')

    # Save
    print(f'\n[5/5] Saving {out_path}')
    with h5py.File(out_path, 'w') as f:
        f.attrs['arc']         = arc
        f.attrs['n_shots']     = N_SHOTS
        f.attrs['sod_mm']      = SOD
        f.attrs['odd_mm']      = ODD
        f.attrs['det_spacing_mm'] = DET_SPACING
        f.attrs['det_rows']    = DET_ROWS
        f.attrs['det_cols']    = DET_COLS
        f.attrs['I0']          = I0
        f.attrs['sigma_s_mm']  = SIGMA_S
        f.attrs['sigma_theta_deg'] = SIGMA_THETA

        sg = f.create_group('shots/nominal')
        sg.create_dataset('source_positions', data=src_positions)
        sg.create_dataset('ground_truth_9dof', data=gt_9dof)
        sg.create_dataset('cone_vec', data=vectors)

        sg2 = f.create_group('sinogram')
        sg2.create_dataset('clean', data=sino_clean, compression='gzip')
        sg2.create_dataset('noisy', data=sino_noisy, compression='gzip')

        cg = f.create_group('centroids')
        cg.create_dataset('positions',       data=positions)
        cg.create_dataset('detection_mask',  data=detection_mask)
        cg.create_dataset('unsharpness_weights', data=weights)

        f.create_dataset('marker_positions', data=marker_3d)

    wall = time.perf_counter() - t0
    print(f'\nDone in {wall:.1f}s — {out_path}')
    print('=' * 60)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--arc', choices=['restricted', 'full360', 'both'], default='both')
    args = p.parse_args()

    arcs = ['restricted', 'full360'] if args.arc == 'both' else [args.arc]
    for arc in arcs:
        run_nih_forward(arc)
