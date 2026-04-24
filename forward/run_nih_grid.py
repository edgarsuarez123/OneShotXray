"""
forward/run_nih_grid.py — NIH Aim 1: Constellation grid + mART sweep.

Constellation grid (6 configs): N_markers=(4,6,8) x arc=(restricted,full360)
  - Subsets detection_mask to simulate fewer fiducials
  - Re-runs SDSG solver for each config
  - Records mean residual per config -> data/nih/aim1_constellation_grid.h5

mART sweep (9 combos): iterations=(25,50,100) x relaxation=(0.5,1.0,2.0)
  - Runs mART on restricted arc with 8 markers
  - Records SSIM, CNR@5mm, convergence curve, wall_time
  - -> data/nih/aim1_mart_sweep.h5

Gate: mART CNR@5mm >= 4 (Rose criterion for hemorrhage detection)
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solver.solver import solve_shot
from recon.mart import reconstruct_mart
from analysis.metrics import compute_ssim, compute_cnr, build_hemorrhage_masks


# ── Grid parameters ───────────────────────────────────────────────────────────
MARKER_COUNTS = [4, 6, 8]
MART_ITERS    = [25, 50, 100]
MART_RELAX    = [0.5, 1.0, 2.0]

NIH_GRID = (200, 160, 140)
NIH_VOXEL_SIZE = 1.0  # mm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_solver_subset(
    sino_path: Path,
    marker_3d: np.ndarray,
    n_markers_use: int,
) -> float:
    """Run SDSG solver using only first n_markers_use detected markers per shot."""
    with h5py.File(sino_path, 'r') as f:
        positions      = f['centroids/positions'][:]
        detection_mask = f['centroids/detection_mask'][:].astype(bool)
        weights        = f['centroids/unsharpness_weights'][:]
        nominal_src    = f['shots/nominal/source_positions'][:]
        det_spacing    = float(f.attrs['det_spacing_mm'])
        det_rows       = int(f.attrs['det_rows'])
        det_cols       = int(f.attrs['det_cols'])
        sod            = float(f.attrs['sod_mm'])
        odd            = float(f.attrs['odd_mm'])

    n_shots, n_markers_total = detection_mask.shape

    rms_vals = []
    for i in range(n_shots):
        # Subset: keep only the first n_markers_use detected markers
        dmask = detection_mask[i].copy()
        detected_indices = np.where(dmask)[0]
        if len(detected_indices) > n_markers_use:
            # Disable markers beyond n_markers_use (last ones by index)
            for idx in detected_indices[n_markers_use:]:
                dmask[idx] = False

        res = solve_shot(
            marker_3d,
            positions[i],
            weights[i],
            dmask,
            det_spacing, det_rows, det_cols, sod, odd,
            nominal_src=nominal_src[i],
        )
        if not np.isnan(res['per_shot_rms']):
            rms_vals.append(res['per_shot_rms'])

    return float(np.mean(rms_vals)) if rms_vals else np.nan


# ── Step 1: Constellation Grid ────────────────────────────────────────────────

def run_constellation_grid(
    restricted_path: Path,
    full360_path: Path,
    phantom_path: Path,
    out_path: Path,
) -> None:
    print('\n' + '=' * 60)
    print('Aim 1 — Constellation Grid')
    print('=' * 60)

    with h5py.File(phantom_path, 'r') as f:
        marker_3d = f['marker_positions'][:].astype(np.float64)

    arc_paths = [
        ('restricted_180', restricted_path),
        ('full_360',       full360_path),
    ]

    results = {}
    for arc_label, sino_path in arc_paths:
        for n_m in MARKER_COUNTS:
            key = f'{arc_label}_m{n_m}'
            print(f'\n  Config: {arc_label}, {n_m} markers ...', end='', flush=True)
            t0 = time.perf_counter()
            mean_res = _run_solver_subset(sino_path, marker_3d, n_m)
            elapsed  = time.perf_counter() - t0
            print(f'  mean_residual={mean_res:.4f}px  ({elapsed:.1f}s)')
            results[key] = mean_res

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        for key, val in results.items():
            f.create_dataset(key, data=np.float64(val))
        # Also store as structured arrays for easy plotting
        arc_labels  = []
        marker_ns   = []
        residuals   = []
        for arc_label, _ in arc_paths:
            for n_m in MARKER_COUNTS:
                arc_labels.append(arc_label.encode())
                marker_ns.append(n_m)
                residuals.append(results[f'{arc_label}_m{n_m}'])
        f.create_dataset('arc_labels',  data=np.array(arc_labels))
        f.create_dataset('marker_ns',   data=np.array(marker_ns))
        f.create_dataset('residuals_px', data=np.array(residuals))

    print(f'\n  Saved constellation grid -> {out_path}')
    print('\n  Summary:')
    for (arc_label, _) in arc_paths:
        row = [f'{results[f"{arc_label}_m{n_m}"]:.4f}px' for n_m in MARKER_COUNTS]
        print(f'    {arc_label:18s} | markers 4={row[0]}  6={row[1]}  8={row[2]}')


# ── Step 2: mART Sweep ────────────────────────────────────────────────────────

def run_mart_sweep(
    restricted_path: Path,
    phantom_path: Path,
    out_path: Path,
) -> float:
    """Run 9-combo mART sweep on restricted arc. Returns best CNR@5mm."""
    print('\n' + '=' * 60)
    print('Aim 1 — mART Sweep (9 combos)')
    print('=' * 60)

    with h5py.File(restricted_path, 'r') as f:
        sinogram = f['sinogram'][:]
        vectors  = f['shots/nominal/cone_vec'][:]

    with h5py.File(phantom_path, 'r') as f:
        phantom_vol  = f['volume'][:].astype(np.float32)
        hemorrhage_c = f.attrs['hemorrhage_center_mm']

    defect_mask, bg_mask = build_hemorrhage_masks(
        hemorrhage_center_mm=np.array(hemorrhage_c),
        lesion_radius_mm=5.0,
        grid_size=NIH_GRID,
        voxel_size=NIH_VOXEL_SIZE,
    )

    n_combos = len(MART_ITERS) * len(MART_RELAX)
    results = {
        'n_iter':        [],
        'relaxation':    [],
        'ssim':          [],
        'cnr_5mm':       [],
        'wall_time_s':   [],
        'convergence':   [],   # list of arrays
    }

    best_cnr  = -np.inf
    gate_pass = False

    combo_idx = 0
    for n_iter in MART_ITERS:
        for relax in MART_RELAX:
            combo_idx += 1
            print(f'\n  [{combo_idx}/{n_combos}] mART n_iter={n_iter}, relaxation={relax:.1f}')
            t0 = time.perf_counter()

            vol, conv = reconstruct_mart(
                sinogram, vectors,
                n_iter=n_iter,
                relaxation=relax,
                voxel_size=NIH_VOXEL_SIZE,
                grid_size=NIH_GRID,
            )

            elapsed = time.perf_counter() - t0

            ssim_val = compute_ssim(vol, phantom_vol)
            cnr_val  = compute_cnr(vol, defect_mask, bg_mask)

            print(f'    SSIM={ssim_val:.4f}  CNR@5mm={cnr_val:.3f}  '
                  f'({"PASS" if cnr_val >= 4.0 else "FAIL"} gate=4)  {elapsed:.1f}s')

            results['n_iter'].append(n_iter)
            results['relaxation'].append(relax)
            results['ssim'].append(ssim_val)
            results['cnr_5mm'].append(cnr_val)
            results['wall_time_s'].append(elapsed)
            results['convergence'].append(conv)

            if cnr_val > best_cnr:
                best_cnr = cnr_val
                if cnr_val >= 4.0:
                    gate_pass = True

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('n_iter',        data=np.array(results['n_iter']))
        f.create_dataset('relaxation',    data=np.array(results['relaxation']))
        f.create_dataset('ssim',          data=np.array(results['ssim']))
        f.create_dataset('cnr_5mm',       data=np.array(results['cnr_5mm']))
        f.create_dataset('wall_time_s',   data=np.array(results['wall_time_s']))
        max_len = max(len(c) for c in results['convergence'])
        conv_arr = np.full((n_combos, max_len), np.nan)
        for k, c in enumerate(results['convergence']):
            conv_arr[k, :len(c)] = c
        f.create_dataset('convergence', data=conv_arr)

    print(f'\n  Saved mART sweep -> {out_path}')
    print(f'\n  Best CNR@5mm = {best_cnr:.3f}  '
          f'({"PASS — gate CNR >= 4" if gate_pass else "FAIL — gate CNR >= 4"})')
    return best_cnr


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data_dir    = ROOT / 'data' / 'nih'
    phantom_path     = data_dir / 'phantom_lesion_5mm.h5'
    restricted_path  = data_dir / 'sinogram_80_restricted.h5'
    full360_path     = data_dir / 'sinogram_80_full360.h5'
    grid_out_path    = data_dir / 'aim1_constellation_grid.h5'
    sweep_out_path   = data_dir / 'aim1_mart_sweep.h5'

    run_constellation_grid(
        restricted_path, full360_path, phantom_path, grid_out_path,
    )

    best_cnr = run_mart_sweep(
        restricted_path, phantom_path, sweep_out_path,
    )

    print('\n=== Aim 1 Summary ===')
    print(f'  Constellation grid saved -> {grid_out_path}')
    print(f'  mART sweep saved         -> {sweep_out_path}')
    print(f'  Best CNR@5mm             = {best_cnr:.3f}  '
          f'({"PASS" if best_cnr >= 4.0 else "FAIL"} — gate >= 4)')

    if best_cnr < 4.0:
        print('\n*** Aim 1 GATE FAIL: mART CNR@5mm < 4 ***')
        sys.exit(1)


if __name__ == '__main__':
    main()
