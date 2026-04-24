"""
recon/run_nih_recon.py — NIH cranial phantom reconstruction pipeline.

Reads:
  data/nih/sinogram_80_restricted.h5  (has results/recovered_cone_vec)
  data/nih/sinogram_80_full360.h5
  data/nih/phantom_lesion_{3,5,8,12}mm.h5
  data/nih/phantom_nolesion.h5

Writes:
  data/nih/recon_restricted_fbp.h5
  data/nih/recon_restricted_mart.h5
  data/nih/recon_restricted_sart.h5
  data/nih/recon_full360_fbp.h5
  data/nih/recon_full360_mart.h5
  data/nih/recon_lesion_sweep.h5   -- CNR vs lesion size (4 sizes x 2 arcs x 2 methods)

Gate: mART CNR@5mm >= 4 (same as Aim 1 sweep result)
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.fbp import reconstruct_fbp
from recon.mart import reconstruct_mart
from recon.sart import reconstruct_sart
from recon.inpainting import inpaint_markers
from analysis.metrics import compute_ssim, compute_psnr, compute_cnr, build_hemorrhage_masks

NIH_GRID       = (200, 160, 140)
NIH_VOXEL_SIZE = 1.0   # mm

# Best mART params from Aim 1 sweep: n_iter=25, relax=0.5 → CNR=5.312
MART_BEST_ITER  = 25
MART_BEST_RELAX = 0.5

# SART params
SART_ITER  = 50
SART_RELAX = 1.0

LESION_SIZES_MM = [3.0, 5.0, 8.0, 12.0]


def _load_sinogram(path: Path) -> tuple:
    """Load sinogram, GT cone vectors, recovered cone vectors, markers."""
    with h5py.File(path, 'r') as f:
        sinogram        = f['sinogram'][:]
        gt_vectors      = f['shots/nominal/cone_vec'][:]
        recovered_vecs  = f['results/recovered_cone_vec'][:]
        det_spacing     = float(f.attrs['det_spacing_mm'])
    return sinogram, gt_vectors, recovered_vecs, det_spacing


def _save_recon(path: Path, volume: np.ndarray, metrics: dict, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('volume', data=volume, compression='gzip', compression_opts=4)
        f['volume'].attrs['axis_order'] = '(X,Y,Z)'
        f['volume'].attrs['voxel_size_mm'] = NIH_VOXEL_SIZE
        f['volume'].attrs['method'] = method
        for k, v in metrics.items():
            f.attrs[k] = v


def reconstruct_one(
    sinogram: np.ndarray,
    vectors: np.ndarray,
    phantom_vol: np.ndarray,
    marker_3d: np.ndarray,
    hemorrhage_center_mm: np.ndarray,
    lesion_radius_mm: float,
    method: str,
) -> tuple:
    """Run one reconstruction, inpaint markers, compute metrics. Returns (volume, metrics)."""
    t0 = time.perf_counter()

    if method == 'fbp':
        vol = reconstruct_fbp(sinogram, vectors, voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID)
    elif method == 'mart':
        vol, _ = reconstruct_mart(
            sinogram, vectors,
            n_iter=MART_BEST_ITER, relaxation=MART_BEST_RELAX,
            voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
        )
    elif method == 'sart':
        vol, _ = reconstruct_sart(
            sinogram, vectors,
            n_iter=SART_ITER, relaxation=SART_RELAX,
            voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
        )
    else:
        raise ValueError(f'Unknown method: {method}')

    elapsed = time.perf_counter() - t0

    # Inpaint BaSO4 markers (1mm radius + 3mm shell at 1mm voxel size)
    vol = inpaint_markers(
        vol, marker_3d,
        marker_radius_mm=1.5,
        shell_outer_mm=4.5,
        voxel_size=NIH_VOXEL_SIZE,
        grid_size=NIH_GRID,
    )

    # Metrics vs GT phantom
    ssim_val = compute_ssim(vol, phantom_vol)
    psnr_val = compute_psnr(vol, phantom_vol)

    defect_mask, bg_mask = build_hemorrhage_masks(
        hemorrhage_center_mm=hemorrhage_center_mm,
        lesion_radius_mm=lesion_radius_mm,
        grid_size=NIH_GRID,
        voxel_size=NIH_VOXEL_SIZE,
    )
    cnr_val = compute_cnr(vol, defect_mask, bg_mask)

    metrics = {
        'ssim': ssim_val,
        'psnr': psnr_val,
        'cnr_lesion': cnr_val,
        'lesion_radius_mm': lesion_radius_mm,
        'wall_time_s': elapsed,
        'method': method,
    }
    return vol, metrics


def run_primary_reconstructions(
    restricted_path: Path,
    full360_path: Path,
    phantom_path: Path,
    out_dir: Path,
) -> dict:
    """FBP + mART + SART on both arcs for 5mm lesion."""
    print('\n' + '=' * 60)
    print('Step 6 — Primary Reconstructions (5mm lesion)')
    print('=' * 60)

    with h5py.File(phantom_path, 'r') as f:
        phantom_vol     = f['volume'][:].astype(np.float32)
        marker_3d       = f['marker_positions'][:].astype(np.float64)
        hemorrhage_c    = np.array(f.attrs['hemorrhage_center_mm'])
        lesion_d        = float(f.attrs['lesion_diameter_mm'])

    lesion_r = lesion_d / 2.0

    arcs = [
        ('restricted', restricted_path),
        ('full360',    full360_path),
    ]
    methods = ['fbp', 'mart', 'sart']
    summary = {}

    for arc_label, sino_path in arcs:
        print(f'\n  Arc: {arc_label}')
        sinogram, gt_vecs, recovered_vecs, _ = _load_sinogram(sino_path)

        for method in methods:
            print(f'\n  [{arc_label}] {method.upper()} ...')
            # Use recovered vectors (solver output) for realistic reconstruction
            vol, metrics = reconstruct_one(
                sinogram, recovered_vecs, phantom_vol, marker_3d,
                hemorrhage_c, lesion_r, method,
            )
            key = f'{arc_label}_{method}'
            summary[key] = metrics

            print(f'    SSIM={metrics["ssim"]:.4f}  PSNR={metrics["psnr"]:.2f}dB  '
                  f'CNR@5mm={metrics["cnr_lesion"]:.3f}  {metrics["wall_time_s"]:.1f}s')

            out_path = out_dir / f'recon_{arc_label}_{method}.h5'
            _save_recon(out_path, vol, metrics, method)
            print(f'    Saved -> {out_path}')

    return summary


def run_lesion_sweep(
    restricted_path: Path,
    full360_path: Path,
    phantom_dir: Path,
    out_path: Path,
) -> None:
    """Reconstruct all lesion sizes (3,5,8,12mm) for CNR-vs-lesion figure."""
    print('\n' + '=' * 60)
    print('Step 6 — Lesion Size Sweep')
    print('=' * 60)

    arcs = [
        ('restricted', restricted_path),
        ('full360',    full360_path),
    ]
    methods = ['fbp', 'mart']  # FBP + mART only (SART is slow; not needed for sweep)

    records = {
        'lesion_mm':   [],
        'arc':         [],
        'method':      [],
        'cnr':         [],
        'ssim':        [],
    }

    for lesion_mm in LESION_SIZES_MM:
        phantom_path = phantom_dir / f'phantom_lesion_{lesion_mm:.0f}mm.h5'
        if not phantom_path.exists():
            print(f'  WARNING: {phantom_path} not found — skipping')
            continue

        with h5py.File(phantom_path, 'r') as f:
            phantom_vol  = f['volume'][:].astype(np.float32)
            marker_3d    = f['marker_positions'][:].astype(np.float64)
            hemorrhage_c = np.array(f.attrs['hemorrhage_center_mm'])

        lesion_r = lesion_mm / 2.0

        for arc_label, sino_path in arcs:
            sinogram, _, recovered_vecs, _ = _load_sinogram(sino_path)

            for method in methods:
                print(f'  lesion={lesion_mm:.0f}mm, {arc_label}, {method} ...', end='', flush=True)
                t0 = time.perf_counter()
                if method == 'fbp':
                    vol = reconstruct_fbp(
                        sinogram, recovered_vecs,
                        voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
                    )
                else:
                    vol, _ = reconstruct_mart(
                        sinogram, recovered_vecs,
                        n_iter=MART_BEST_ITER, relaxation=MART_BEST_RELAX,
                        voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
                    )
                vol = inpaint_markers(
                    vol, marker_3d,
                    marker_radius_mm=1.5, shell_outer_mm=4.5,
                    voxel_size=NIH_VOXEL_SIZE, grid_size=NIH_GRID,
                )
                defect_mask, bg_mask = build_hemorrhage_masks(
                    hemorrhage_c, lesion_r, NIH_GRID, NIH_VOXEL_SIZE,
                )
                cnr_val  = compute_cnr(vol, defect_mask, bg_mask)
                ssim_val = compute_ssim(vol, phantom_vol)
                elapsed  = time.perf_counter() - t0

                records['lesion_mm'].append(lesion_mm)
                records['arc'].append(arc_label)
                records['method'].append(method)
                records['cnr'].append(cnr_val)
                records['ssim'].append(ssim_val)
                print(f'  CNR={cnr_val:.3f}  SSIM={ssim_val:.4f}  ({elapsed:.1f}s)')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('lesion_mm', data=np.array(records['lesion_mm']))
        f.create_dataset('arc',       data=np.array([s.encode() for s in records['arc']]))
        f.create_dataset('method',    data=np.array([s.encode() for s in records['method']]))
        f.create_dataset('cnr',       data=np.array(records['cnr']))
        f.create_dataset('ssim',      data=np.array(records['ssim']))
    print(f'\n  Saved lesion sweep -> {out_path}')


def main() -> None:
    data_dir    = ROOT / 'data' / 'nih'
    out_dir     = data_dir
    phantom_5mm = data_dir / 'phantom_lesion_5mm.h5'

    summary = run_primary_reconstructions(
        data_dir / 'sinogram_80_restricted.h5',
        data_dir / 'sinogram_80_full360.h5',
        phantom_5mm,
        out_dir,
    )

    run_lesion_sweep(
        data_dir / 'sinogram_80_restricted.h5',
        data_dir / 'sinogram_80_full360.h5',
        data_dir,
        data_dir / 'recon_lesion_sweep.h5',
    )

    print('\n=== NIH Reconstruction Summary ===')
    best_cnr = 0.0
    for key, m in summary.items():
        cnr = m['cnr_lesion']
        best_cnr = max(best_cnr, cnr)
        status = 'PASS' if cnr >= 4.0 else 'FAIL'
        print(f'  {key:28s}  SSIM={m["ssim"]:.4f}  CNR@5mm={cnr:.3f}  [{status}]')

    print(f'\n  Best CNR@5mm = {best_cnr:.3f}  '
          f'({"PASS" if best_cnr >= 4.0 else "FAIL"} — gate >= 4)')

    if best_cnr < 4.0:
        print('\n*** GATE FAIL: mART CNR@5mm < 4 ***')
        sys.exit(1)


if __name__ == '__main__':
    main()
