"""
recon/run_nih_recon.py — NIH reconstruction orchestrator.

Runs FBP, mART (25 iter, lam=0.5), and SART (50 iter, lam=1.0) on both arcs.
Also runs the Aim 1 constellation grid and mART parameter sweep.
All outputs -> data/nih/.

Gate: restricted mART CNR@5mm >= 4.0 (Rose criterion).
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recon.mart import reconstruct_mart
from recon.sart import reconstruct_sart
from recon.fbp  import reconstruct_fbp
from analysis.metrics import compute_ssim, compute_psnr, compute_cnr
from analysis.metrics import build_nih_hemorrhage_mask, build_nih_bg_mask
from phantom.nih_phantom import NX, NY, NZ, VOXEL_SIZE, DEFAULT_LESION_MM
from forward.nih_geometry import DET_SPACING, DET_ROWS, DET_COLS


MART_N_ITER  = 25
MART_LAM     = 0.5
SART_N_ITER  = 50
SART_LAM     = 1.0
ROSE_CNR     = 4.0


def _load_sinogram(arc: str) -> dict:
    path = ROOT / 'data' / 'nih' / f'sinogram_80_{arc}.h5'
    with h5py.File(path, 'r') as f:
        sino   = f['sinogram/noisy'][:]
        vecs   = f['results/recovered_cone_vec'][:]
        marker = f['marker_positions'][:]
    return {'sino': sino, 'vecs': vecs, 'marker_3d': marker}


def _load_phantom(tag: str) -> np.ndarray:
    path = ROOT / 'data' / 'nih' / f'phantom_{tag}.h5'
    with h5py.File(path, 'r') as f:
        return f['volume'][:]


def _nih_masks():
    return build_nih_hemorrhage_mask(), build_nih_bg_mask()


def _metrics(recon, ref, hem_mask, bg_mask):
    return {
        'ssim': compute_ssim(recon, ref),
        'psnr': compute_psnr(recon, ref),
        'cnr':  compute_cnr(recon, hem_mask, bg_mask),
    }


def run_primary_reconstructions() -> dict:
    """FBP + mART + SART on restricted and full360 arcs, 5mm lesion."""
    results = {}
    phantom_ref = _load_phantom('lesion_5mm')
    hem_mask, bg_mask = _nih_masks()

    for arc in ('restricted', 'full360'):
        d = _load_sinogram(arc)
        sino, vecs = d['sino'], d['vecs']

        # FBP
        print(f'\n--- FBP {arc} ---')
        fbp_vol = reconstruct_fbp(sino, vecs, voxel_size=VOXEL_SIZE,
                                  grid_nx=NX, grid_ny=NY, grid_nz=NZ)
        fbp_m = _metrics(fbp_vol, phantom_ref, hem_mask, bg_mask)

        # mART
        print(f'\n--- mART {arc} (n={MART_N_ITER}, lam={MART_LAM}) ---')
        mart_vol, _ = reconstruct_mart(
            sino, vecs, n_iter=MART_N_ITER, lam=MART_LAM,
            voxel_size=VOXEL_SIZE, grid_nx=NX, grid_ny=NY, grid_nz=NZ,
        )
        mart_m = _metrics(mart_vol, phantom_ref, hem_mask, bg_mask)

        # SART
        print(f'\n--- SART {arc} (n={SART_N_ITER}, lam={SART_LAM}) ---')
        sart_vol, _ = reconstruct_sart(
            sino, vecs, n_iter=SART_N_ITER, lam=SART_LAM,
            voxel_size=VOXEL_SIZE, grid_nx=NX, grid_ny=NY, grid_nz=NZ,
        )
        sart_m = _metrics(sart_vol, phantom_ref, hem_mask, bg_mask)

        results[arc] = {
            'fbp': {'vol': fbp_vol, **fbp_m},
            'mart': {'vol': mart_vol, **mart_m},
            'sart': {'vol': sart_vol, **sart_m},
        }
        print(f'\n{arc}:')
        for method, m in [('FBP', fbp_m), ('mART', mart_m), ('SART', sart_m)]:
            gate = '' if method != 'mART' or arc != 'restricted' else \
                   f'  {"PASS" if m["cnr"] >= ROSE_CNR else "FAIL"} (gate >= {ROSE_CNR})'
            print(f'  {method}: SSIM={m["ssim"]:.3f}  PSNR={m["psnr"]:.2f}dB  '
                  f'CNR@5mm={m["cnr"]:.3f}{gate}')

        # Save volumes
        out = ROOT / 'data' / 'nih' / f'recon_{arc}.h5'
        with h5py.File(out, 'w') as f:
            f.attrs['arc'] = arc
            for method in ('fbp', 'mart', 'sart'):
                g = f.create_group(method)
                g.create_dataset('volume', data=results[arc][method]['vol'], compression='gzip')
                for k in ('ssim', 'psnr', 'cnr'):
                    g.attrs[k] = results[arc][method][k]
        print(f'  Saved {out}')

    # Gate check
    cnr_gate = results['restricted']['mart']['cnr']
    if cnr_gate < ROSE_CNR:
        print(f'\n*** GATE FAIL: restricted mART CNR@5mm={cnr_gate:.3f} < {ROSE_CNR} ***')
        sys.exit(1)
    else:
        print(f'\nGATE PASS: restricted mART CNR@5mm={cnr_gate:.3f} >= {ROSE_CNR}')

    return results


def run_lesion_sweep() -> dict:
    """CNR vs lesion size: {3,5,8,12}mm x {restricted,full360} x mART."""
    lesions = [3.0, 5.0, 8.0, 12.0]
    results = {}
    hem_mask, bg_mask = _nih_masks()

    for arc in ('restricted', 'full360'):
        d = _load_sinogram(arc)
        sino, vecs = d['sino'], d['vecs']
        results[arc] = {}

        for d_mm in lesions:
            tag = f'lesion_{int(d_mm)}mm' if d_mm > 0 else 'nolesion'
            phantom_ref = _load_phantom(tag)

            print(f'\n--- mART {arc} {d_mm}mm ---')
            vol, _ = reconstruct_mart(
                sino, vecs, n_iter=MART_N_ITER, lam=MART_LAM,
                voxel_size=VOXEL_SIZE, grid_nx=NX, grid_ny=NY, grid_nz=NZ,
            )
            m = _metrics(vol, phantom_ref, hem_mask, bg_mask)
            results[arc][d_mm] = m
            print(f'  CNR={m["cnr"]:.3f}  SSIM={m["ssim"]:.3f}')

    out = ROOT / 'data' / 'nih' / 'recon_lesion_sweep.h5'
    with h5py.File(out, 'w') as f:
        for arc in results:
            g = f.create_group(arc)
            for d_mm, m in results[arc].items():
                sg = g.create_group(f'lesion_{d_mm}mm')
                for k, v in m.items():
                    sg.attrs[k] = v
    print(f'\nLesion sweep saved -> {out}')
    return results


def run_aim1_constellation_grid() -> np.ndarray:
    """Solver residuals for {4,6,8} markers x {restricted,full360}."""
    from solver.solver import solve_shot
    from solver.projection import params_to_cone_vec

    marker_counts = [4, 6, 8]
    results = {}
    hem_mask, bg_mask = _nih_masks()

    for arc in ('restricted', 'full360'):
        sino_path = ROOT / 'data' / 'nih' / f'sinogram_80_{arc}.h5'
        with h5py.File(sino_path, 'r') as f:
            positions      = f['centroids/positions'][:]
            detection_mask = f['centroids/detection_mask'][:].astype(bool)
            weights        = f['centroids/unsharpness_weights'][:]
            nominal_src    = f['shots/nominal/source_positions'][:]
            gt_9dof        = f['shots/nominal/ground_truth_9dof'][:]
            marker_3d      = f['marker_positions'][:].astype(np.float64)
            det_spacing    = float(f.attrs['det_spacing_mm'])
            det_rows       = int(f.attrs['det_rows'])
            det_cols       = int(f.attrs['det_cols'])
            sod            = float(f.attrs['sod_mm'])
            odd            = float(f.attrs['odd_mm'])

        n_shots, n_markers, _ = positions.shape
        results[arc] = {}

        for n_m in marker_counts:
            # Only keep the first n_m markers
            mask_trunc = detection_mask.copy()
            mask_trunc[:, n_m:] = False

            rms_list = []
            for i in range(n_shots):
                res = solve_shot(
                    marker_3d[:n_m], positions[i, :n_m], weights[i, :n_m],
                    mask_trunc[i, :n_m],
                    det_spacing, det_rows, det_cols, sod, odd,
                    nominal_src=nominal_src[i],
                )
                if not np.isnan(res['per_shot_rms']):
                    rms_list.append(res['per_shot_rms'])

            mean_res = float(np.mean(rms_list)) if rms_list else np.nan
            results[arc][n_m] = mean_res
            print(f'  {arc} {n_m} markers: mean residual = {mean_res:.4f} px')

    out = ROOT / 'data' / 'nih' / 'aim1_constellation_grid.h5'
    with h5py.File(out, 'w') as f:
        for arc in results:
            g = f.create_group(arc)
            for n_m, res in results[arc].items():
                g.attrs[f'markers_{n_m}_mean_residual_px'] = res
    print(f'\nConstellation grid saved -> {out}')
    return results


def run_aim1_mart_sweep() -> dict:
    """mART convergence sweep: {25,50,100} iter x {0.5,1.0,2.0} lam."""
    iterations_list = [25, 50, 100]
    lam_list        = [0.5, 1.0, 2.0]

    hem_mask, bg_mask = _nih_masks()
    phantom_ref = _load_phantom('lesion_5mm')
    d = _load_sinogram('restricted')
    sino, vecs = d['sino'], d['vecs']

    results = {}
    for n_iter in iterations_list:
        results[n_iter] = {}
        for lam in lam_list:
            print(f'\n--- mART sweep n_iter={n_iter} lam={lam} ---')
            vol, conv = reconstruct_mart(
                sino, vecs, n_iter=n_iter, lam=lam,
                voxel_size=VOXEL_SIZE, grid_nx=NX, grid_ny=NY, grid_nz=NZ,
            )
            cnr = compute_cnr(vol, hem_mask, bg_mask)
            ssim = compute_ssim(vol, phantom_ref)
            results[n_iter][lam] = {'cnr': cnr, 'ssim': ssim, 'convergence': conv}
            print(f'  CNR={cnr:.3f}  SSIM={ssim:.3f}')

    out = ROOT / 'data' / 'nih' / 'aim1_mart_sweep.h5'
    with h5py.File(out, 'w') as f:
        for n_iter, lam_dict in results.items():
            g = f.create_group(f'iter_{n_iter}')
            for lam, m in lam_dict.items():
                sg = g.create_group(f'lam_{lam}')
                sg.create_dataset('convergence', data=m['convergence'])
                sg.attrs['cnr']  = m['cnr']
                sg.attrs['ssim'] = m['ssim']
    print(f'\nmART sweep saved -> {out}')
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--step', choices=['primary', 'lesion', 'aim1_grid', 'aim1_sweep', 'all'],
                   default='all')
    args = p.parse_args()

    t0 = time.perf_counter()
    if args.step in ('primary', 'all'):
        run_primary_reconstructions()
    if args.step in ('lesion', 'all'):
        run_lesion_sweep()
    if args.step in ('aim1_grid', 'all'):
        run_aim1_constellation_grid()
    if args.step in ('aim1_sweep', 'all'):
        run_aim1_mart_sweep()
    print(f'\nTotal wall time: {time.perf_counter() - t0:.1f}s')
