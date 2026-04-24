"""
recon/run_navy_recon.py — Navy reconstruction pipeline orchestrator.

Pipeline:
  1. Load phantom + sinogram_100.h5 (sinogram + recovered_cone_vec)
  2. FBP (FDK_CUDA) reconstruction
  3. mART (50 iter) reconstruction
  4. Inpaint fiducial markers on FBP, mART, and reference phantom
  5. Compute metrics (SSIM, PSNR, CNR@0.4/0.8/1.6mm)
  6. GATE: mART CNR@0.8mm >= 4 (Rose criterion)
  7. Save results to sinogram_100.h5 under 'recon/' group
  8. Print summary table

Gate fallback order (if CNR@0.8mm < 4):
  1. Increase mART to 100 iterations
  2. Add TV denoising (denoise_tv_chambolle, weight=0.01) after mART
"""

import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SINO_PATH    = ROOT / 'data' / 'navy' / 'sinogram_100.h5'
PHANTOM_PATH = ROOT / 'data' / 'navy' / 'phantom.h5'


def run_pipeline(n_iter_mart: int = 50, tv_denoise: bool = False) -> dict:
    """
    Run FBP + mART + inpainting + metrics.

    Returns
    -------
    results dict with keys: vol_fbp, vol_mart, vol_ref, metrics
    """
    from recon.fbp      import reconstruct_fbp
    from recon.mart     import reconstruct_mart
    from recon.inpainting import inpaint_markers
    from analysis.metrics  import compute_ssim, compute_psnr, compute_cnr, build_crack_masks

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print('\n[1/7] Loading phantom and sinogram...')
    with h5py.File(PHANTOM_PATH, 'r') as f:
        phantom       = f['volume'][:].astype(np.float32)          # (250,250,250)
        marker_pos_mm = f['marker_positions'][:].astype(np.float64)  # (8, 3)
        voxel_size    = float(f.attrs['voxel_size_mm'])

    with h5py.File(SINO_PATH, 'r') as f:
        sinogram      = f['sinogram'][:]                          # (512,100,512)
        recovered_vec = f['results/recovered_cone_vec'][:]        # (100,12)

    grid_size = phantom.shape[0]
    print(f'      phantom: {phantom.shape}, voxel={voxel_size}mm')
    print(f'      sinogram: {sinogram.shape}')
    n_nan = int(np.sum(np.any(np.isnan(recovered_vec), axis=1)))
    print(f'      recovered_cone_vec: {recovered_vec.shape} ({n_nan} NaN rows)')

    # ── 2. FBP ────────────────────────────────────────────────────────────────
    print('\n[2/7] FBP reconstruction (FDK_CUDA)...')
    t0 = time.perf_counter()
    vol_fbp = reconstruct_fbp(sinogram, recovered_vec, voxel_size, grid_size)
    print(f'      Done in {time.perf_counter()-t0:.1f}s  '
          f'range [{vol_fbp.min():.4f}, {vol_fbp.max():.4f}]')

    # ── 3. mART ───────────────────────────────────────────────────────────────
    print(f'\n[3/7] mART reconstruction ({n_iter_mart} iter)...')
    t0 = time.perf_counter()
    vol_mart, convergence = reconstruct_mart(
        sinogram, recovered_vec,
        n_iter=n_iter_mart, voxel_size=voxel_size, grid_size=grid_size,
    )
    print(f'      Done in {time.perf_counter()-t0:.1f}s  '
          f'range [{vol_mart.min():.4f}, {vol_mart.max():.4f}]')
    print(f'      Final convergence norm: {convergence[-1]:.6f}')

    # Optional TV denoising
    if tv_denoise:
        print('      Applying TV denoising (weight=0.01)...')
        from skimage.restoration import denoise_tv_chambolle
        vol_mart = denoise_tv_chambolle(vol_mart, weight=0.01, channel_axis=None).astype(np.float32)
        np.clip(vol_mart, 0.0, None, out=vol_mart)

    # ── 4. Inpainting ─────────────────────────────────────────────────────────
    print('\n[4/7] Inpainting fiducial markers...')
    vol_fbp_inp  = inpaint_markers(vol_fbp,  marker_pos_mm, voxel_size=voxel_size, grid_size=grid_size)
    vol_mart_inp = inpaint_markers(vol_mart, marker_pos_mm, voxel_size=voxel_size, grid_size=grid_size)
    phantom_inp  = inpaint_markers(phantom,  marker_pos_mm, voxel_size=voxel_size, grid_size=grid_size)
    print('      FBP, mART, and phantom inpainted.')

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    print('\n[5/7] Computing metrics...')
    crack_masks = build_crack_masks(grid_size, voxel_size)

    ssim_fbp  = compute_ssim(vol_fbp_inp,  phantom_inp)
    ssim_mart = compute_ssim(vol_mart_inp, phantom_inp)
    psnr_fbp  = compute_psnr(vol_fbp_inp,  phantom_inp)
    psnr_mart = compute_psnr(vol_mart_inp, phantom_inp)

    widths_mm = [0.4, 0.8, 1.6]
    cnr_fbp   = []
    cnr_mart  = []
    for w in widths_mm:
        dm, bm = crack_masks[w]
        cnr_fbp.append(compute_cnr(vol_fbp_inp,  dm, bm))
        cnr_mart.append(compute_cnr(vol_mart_inp, dm, bm))

    print(f'      SSIM  FBP={ssim_fbp:.4f}  mART={ssim_mart:.4f}')
    print(f'      PSNR  FBP={psnr_fbp:.2f}dB  mART={psnr_mart:.2f}dB')
    for i, w in enumerate(widths_mm):
        print(f'      CNR@{w}mm  FBP={cnr_fbp[i]:.2f}  mART={cnr_mart[i]:.2f}')

    metrics = {
        'ssim_fbp': ssim_fbp, 'ssim_mart': ssim_mart,
        'psnr_fbp': psnr_fbp, 'psnr_mart': psnr_mart,
        'cnr_fbp':  np.array(cnr_fbp),
        'cnr_mart': np.array(cnr_mart),
        'crack_widths_mm': np.array(widths_mm),
    }

    return {
        'vol_fbp':  vol_fbp_inp,
        'vol_mart': vol_mart_inp,
        'vol_ref':  phantom_inp,
        'convergence': convergence,
        'metrics':  metrics,
    }


def save_results(results: dict) -> None:
    """Save reconstruction volumes and metrics to sinogram_100.h5 under 'recon/' group."""
    print(f'\n[6/7] Saving results to {SINO_PATH}...')
    with h5py.File(SINO_PATH, 'a') as f:
        if 'recon' in f:
            del f['recon']
        g = f.create_group('recon')

        g.create_dataset('vol_fbp_inpainted',  data=results['vol_fbp'],  compression='gzip', compression_opts=4)
        g.create_dataset('vol_mart_inpainted', data=results['vol_mart'], compression='gzip', compression_opts=4)
        g.create_dataset('phantom_inpainted',  data=results['vol_ref'],  compression='gzip', compression_opts=4)
        g.create_dataset('convergence_mart',   data=results['convergence'])

        mg = g.create_group('metrics')
        m  = results['metrics']
        mg.create_dataset('crack_widths_mm', data=m['crack_widths_mm'])
        mg.create_dataset('cnr_fbp',         data=m['cnr_fbp'])
        mg.create_dataset('cnr_mart',        data=m['cnr_mart'])
        for key in ('ssim_fbp', 'ssim_mart', 'psnr_fbp', 'psnr_mart'):
            mg.attrs[key] = m[key]
            # Also as dataset for easy loading
            mg.create_dataset(key, data=np.float64(m[key]))

    print('      Saved.')


def main() -> None:
    t_total = time.perf_counter()
    print('=' * 60)
    print('Navy Reconstruction Pipeline — FBP + mART')
    print('=' * 60)

    # First attempt: 50 iterations
    results = run_pipeline(n_iter_mart=50, tv_denoise=False)
    cnr_08  = float(results['metrics']['cnr_mart'][1])

    # Gate check
    print(f'\n[GATE] mART CNR@0.8mm = {cnr_08:.2f} (target >= 4.0)')

    if cnr_08 < 4.0:
        print('  Gate FAIL — trying 100 iterations...')
        results = run_pipeline(n_iter_mart=100, tv_denoise=False)
        cnr_08  = float(results['metrics']['cnr_mart'][1])
        print(f'  After 100 iter: CNR@0.8mm = {cnr_08:.2f}')

    if cnr_08 < 4.0:
        print('  Still FAIL — adding TV denoising...')
        results = run_pipeline(n_iter_mart=100, tv_denoise=True)
        cnr_08  = float(results['metrics']['cnr_mart'][1])
        print(f'  After TV denoise: CNR@0.8mm = {cnr_08:.2f}')

    # Save
    save_results(results)

    # Print summary table
    m = results['metrics']
    wall = time.perf_counter() - t_total
    print(f'\n{"=" * 60}')
    print(f'Reconstruction complete in {wall:.1f}s')
    print(f'\nMethod   SSIM    PSNR(dB)  CNR@0.4  CNR@0.8  CNR@1.6')
    print(f"FBP      {m['ssim_fbp']:.4f}  {m['psnr_fbp']:7.2f}   {m['cnr_fbp'][0]:6.2f}   {m['cnr_fbp'][1]:6.2f}   {m['cnr_fbp'][2]:6.2f}")
    print(f"mART     {m['ssim_mart']:.4f}  {m['psnr_mart']:7.2f}   {m['cnr_mart'][0]:6.2f}   {m['cnr_mart'][1]:6.2f}   {m['cnr_mart'][2]:6.2f}")
    print(f'\nGATE: mART CNR@0.8mm = {cnr_08:.2f}  '
          f'{"PASS ✓" if cnr_08 >= 4.0 else "FAIL ✗ — check variants or parameters"}')
    print('=' * 60)

    if cnr_08 < 4.0:
        sys.exit(1)


if __name__ == '__main__':
    main()
