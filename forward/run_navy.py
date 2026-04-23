"""
forward/run_navy.py — End-to-end Navy 100-shot forward projection + centroiding pipeline.

Produces: data/navy/sinogram_100.h5

HDF5 layout:
  sinogram                         (512, 100, 512) float32  — noisy sinogram
  shots/nominal/ground_truth_9dof  (100, 9)        float64  — [src_xyz, det_xyz, euler_abc]
  shots/nominal/source_positions   (100, 3)        float64  — nominal (unperturbed) source positions
  shots/nominal/cone_vec           (100, 12)       float64  — ASTRA cone_vec vectors
  centroids/positions              (100, 8, 2)     float64  — refined centroid [row, col] per shot/marker
  centroids/ground_truth_2d        (100, 8, 2)     float64  — reprojected GT positions
  centroids/noise_per_shot         (100,)          float64  — per-shot RMS centroiding error (pixels)
  centroids/unsharpness_weights    (100, 8)        float64  — per-shot/marker weights
  centroids/detection_mask         (100, 8)        bool     — True where marker was detected

Sinogram axis order: (det_rows=512, N_shots=100, det_cols=512)  -- FWD-008
"""

import time
from pathlib import Path

import h5py
import numpy as np

# Project root (two levels up from this file)
ROOT = Path(__file__).resolve().parent.parent

# ── Geometry parameters (CLAUDE.md canonical values) ─────────────────────────
SOD          = 500.0    # mm — source-to-object distance
ODD          = 200.0    # mm — object-to-detector distance
DET_ROWS     = 512
DET_COLS     = 512
DET_SPACING  = 0.2     # mm — detector pixel pitch
FOCAL_SPOT   = 0.5     # mm — focal spot FWHM
N_SHOTS      = 100
SIGMA_S      = 2.0     # mm — nominal position uncertainty
SIGMA_THETA  = 1.0     # deg — nominal orientation uncertainty
I0           = 10_000  # photons — source intensity
SEED         = 42


def main() -> None:
    from forward.geometry import generate_hemisphere_shots, perturb_geometry, geometry_to_cone_vec
    from forward.projector import forward_project, apply_noise_pipeline
    from forward.centroiding import process_all_shots

    t0 = time.perf_counter()
    print("=" * 60)
    print("Navy forward projection pipeline — 100 shots")
    print("=" * 60)

    # ── Step 1: Load phantom ──────────────────────────────────────────────────
    phantom_path = ROOT / 'data' / 'navy' / 'phantom.h5'
    print(f"\n[1/7] Loading phantom from {phantom_path}")
    with h5py.File(phantom_path, 'r') as f:
        volume         = f['volume'][:]           # (250, 250, 250) float32, axis=(X,Y,Z)
        marker_3d      = f['marker_positions'][:] # (8, 3) float32, mm
        voxel_size     = float(f.attrs['voxel_size_mm'])
    print(f"      volume: {volume.shape} {volume.dtype}  "
          f"range [{volume.min():.4f}, {volume.max():.4f}] mm^-1")
    print(f"      markers: {marker_3d.shape}  voxel_size: {voxel_size}mm")

    # ── Step 2: Generate hemisphere shot positions ────────────────────────────
    print(f"\n[2/7] Generating {N_SHOTS} Fibonacci hemisphere shots (seed={SEED})")
    source_positions = generate_hemisphere_shots(N_SHOTS, SOD, seed=SEED)  # (100, 3)
    print(f"      source Z range: [{source_positions[:,2].min():.1f}, "
          f"{source_positions[:,2].max():.1f}] mm")
    print(f"      |src| range:    [{np.linalg.norm(source_positions, axis=1).min():.2f}, "
          f"{np.linalg.norm(source_positions, axis=1).max():.2f}] mm (expected {SOD})")

    # ── Step 3: Perturb geometry (nominal uncertainty) ────────────────────────
    print(f"\n[3/7] Perturbing geometry: sigma_s={SIGMA_S}mm, sigma_theta={SIGMA_THETA}deg")
    gt_9dof = perturb_geometry(
        source_positions, SOD, ODD,
        sigma_s=SIGMA_S, sigma_theta=SIGMA_THETA, seed=SEED,
    )  # (100, 9)

    # Convert to ASTRA cone_vec
    vectors = geometry_to_cone_vec(gt_9dof, DET_SPACING, DET_ROWS, DET_COLS)  # (100, 12)
    print(f"      gt_9dof shape: {gt_9dof.shape}  vectors shape: {vectors.shape}")

    # Sanity check: source position perturbation
    src_delta = np.linalg.norm(gt_9dof[:, :3] - source_positions, axis=1)
    print(f"      source perturbation: mean {src_delta.mean():.3f}mm, "
          f"max {src_delta.max():.3f}mm (expected ~{SIGMA_S}mm)")

    # ── Step 4: Forward project ───────────────────────────────────────────────
    print(f"\n[4/7] Forward projection (ASTRA FP3D_CUDA)...")
    t_fp = time.perf_counter()
    sinogram_clean = forward_project(volume, vectors, voxel_size, DET_ROWS, DET_COLS)
    t_fp = time.perf_counter() - t_fp
    print(f"      Done in {t_fp:.1f}s.  shape: {sinogram_clean.shape}")

    # Sanity check: central ray through ~25mm steel should give ~2.87
    central_val = float(sinogram_clean[DET_ROWS//2, 0, DET_COLS//2])
    print(f"      Central pixel (shot 0): {central_val:.3f} (expected ~2.87 for 25mm steel)")
    if not (1.0 < central_val < 6.0):
        print(f"      WARNING: central sinogram value {central_val:.3f} outside expected range "
              f"[1.0, 6.0] — check geometry units")

    # ── Step 5: Apply noise pipeline ─────────────────────────────────────────
    print(f"\n[5/7] Applying noise pipeline (I0={I0}, focal_spot={FOCAL_SPOT}mm)...")
    t_noise = time.perf_counter()
    sinogram_noisy = apply_noise_pipeline(
        sinogram_clean, I0=I0, focal_spot=FOCAL_SPOT,
        sod=SOD, odd=ODD, det_spacing=DET_SPACING, seed=SEED,
    )
    t_noise = time.perf_counter() - t_noise
    print(f"      Done in {t_noise:.1f}s.  "
          f"range [{sinogram_noisy.min():.3f}, {sinogram_noisy.max():.4f}]")

    # ── Step 6: Centroiding on all shots ──────────────────────────────────────
    print(f"\n[6/7] Running centroiding on {N_SHOTS} shots × {len(marker_3d)} markers...")
    t_cent = time.perf_counter()
    centroid_result = process_all_shots(
        sinogram_noisy, gt_9dof, vectors,
        marker_3d.astype(np.float64),
        det_spacing=DET_SPACING,
        focal_spot_mm=FOCAL_SPOT,
    )
    t_cent = time.perf_counter() - t_cent
    print(f"      Done in {t_cent:.1f}s.")

    # QA metrics
    n_possible  = N_SHOTS * len(marker_3d)
    n_detected  = centroid_result['n_detected']
    detect_rate = n_detected / n_possible * 100.0
    noise       = centroid_result['noise_per_shot']
    valid_noise = noise[~np.isnan(noise)]

    print(f"\n      Detection rate : {n_detected}/{n_possible} = {detect_rate:.1f}%"
          f"  (target >= 95%)")
    if len(valid_noise) > 0:
        print(f"      Centroid noise : mean {valid_noise.mean():.4f}px, "
              f"max {valid_noise.max():.4f}px, "
              f"RMS {np.sqrt(np.mean(valid_noise**2)):.4f}px  (target < 0.15px)")
    else:
        print("      WARNING: No valid noise measurements — zero detections?")

    n_full = int((centroid_result['detection_mask'].sum(axis=1) == len(marker_3d)).sum())
    print(f"      Shots with all {len(marker_3d)} markers detected: {n_full}/{N_SHOTS}")

    # Warn on failures
    if detect_rate < 95.0:
        print("      *** FAIL: Detection rate < 95% (CENT-002) ***")
    if len(valid_noise) > 0 and valid_noise.mean() >= 0.15:
        print("      *** FAIL: Mean centroiding noise >= 0.15px (CENT-005) ***")

    # ── Step 7: Save to HDF5 ─────────────────────────────────────────────────
    out_path = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
    print(f"\n[7/7] Saving to {out_path}...")
    with h5py.File(out_path, 'w') as f:
        # Sinogram — (512, 100, 512) float32
        f.create_dataset('sinogram', data=sinogram_noisy,
                         compression='gzip', compression_opts=4)
        f['sinogram'].attrs['axis_order'] = '(det_rows, N_shots, det_cols)'
        f['sinogram'].attrs['units']      = 'dimensionless line integral'

        # Shot geometry
        grp = f.create_group('shots/nominal')
        grp.create_dataset('ground_truth_9dof',  data=gt_9dof)
        grp.create_dataset('source_positions',   data=source_positions)
        grp.create_dataset('cone_vec',           data=vectors)
        grp['ground_truth_9dof'].attrs['columns'] = (
            'src_x,src_y,src_z,det_x,det_y,det_z,euler_a_deg,euler_b_deg,euler_c_deg'
        )

        # Centroids
        cgrp = f.create_group('centroids')
        cgrp.create_dataset('positions',           data=centroid_result['positions'])
        cgrp.create_dataset('ground_truth_2d',    data=centroid_result['ground_truth_2d'])
        cgrp.create_dataset('noise_per_shot',      data=centroid_result['noise_per_shot'])
        cgrp.create_dataset('unsharpness_weights', data=centroid_result['unsharpness_weights'])
        cgrp.create_dataset('detection_mask',      data=centroid_result['detection_mask'])
        cgrp['positions'].attrs['columns']         = '[row, col] pixel coords'
        cgrp['noise_per_shot'].attrs['units']      = 'pixels RMS'
        cgrp['noise_per_shot'].attrs['target']     = '< 0.15 px (CENT-005)'

        # Metadata
        f.attrs.update({
            'n_shots'      : N_SHOTS,
            'sod_mm'       : SOD,
            'odd_mm'       : ODD,
            'det_rows'     : DET_ROWS,
            'det_cols'     : DET_COLS,
            'det_spacing_mm': DET_SPACING,
            'I0'           : I0,
            'focal_spot_mm': FOCAL_SPOT,
            'sigma_s_mm'   : SIGMA_S,
            'sigma_theta_deg': SIGMA_THETA,
            'seed'         : SEED,
            'voxel_size_mm': voxel_size,
        })

    size_mb = out_path.stat().st_size / 1e6
    print(f"      Saved {size_mb:.1f} MB")

    # ── Final summary ─────────────────────────────────────────────────────────
    total = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total:.1f}s")
    ok_sino = sinogram_noisy.shape == (DET_ROWS, N_SHOTS, DET_COLS)
    ok_9dof = gt_9dof.shape == (N_SHOTS, 9)
    print(f"  sinogram shape   : {sinogram_noisy.shape}  {'OK' if ok_sino else '*** WRONG ***'}")
    print(f"  gt_9dof shape    : {gt_9dof.shape}  {'OK' if ok_9dof else '*** WRONG ***'}")
    print(f"  detection rate   : {detect_rate:.1f}%  {'OK' if detect_rate >= 95.0 else '*** FAIL ***'}")
    if len(valid_noise) > 0:
        print(f"  centroid RMS     : {valid_noise.mean():.4f}px  {'OK' if valid_noise.mean() < 0.15 else '*** FAIL ***'}")
    print(f"  output           : {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT))
    main()
