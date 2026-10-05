"""Diagnostic: check NIH centroid GT positions and marker signal quality."""
import sys
import numpy as np
import h5py
ROOT = 'C:/Users/Edgar/OneShotXRay'
sys.path.insert(0, ROOT)

with h5py.File('data/nih/sinogram_80_restricted.h5', 'r') as f:
    sino = f['sinogram'][:]
    vectors = f['shots/nominal/cone_vec'][:]
    gt_2d = f['centroids/ground_truth_2d'][:]
    positions = f['centroids/positions'][:]
    det_mask = f['centroids/detection_mask'][:]
    noise = f['centroids/noise_per_shot'][:]

print('Shot 0 GT positions (row, col):')
for j, pos in enumerate(gt_2d[0]):
    print(f'  Marker {j}: ({pos[0]:.1f}, {pos[1]:.1f})')

print()
print(f'Shot 0 refined positions (detected={det_mask[0].sum()}):')
for j, pos in enumerate(positions[0]):
    gt = gt_2d[0, j]
    err = np.linalg.norm(pos - gt) if not np.isnan(pos[0]) else np.nan
    print(f'  Marker {j}: ({pos[0]:.1f}, {pos[1]:.1f})  GT=({gt[0]:.1f},{gt[1]:.1f})  err={err:.3f}px')

print()
print(f'Sinogram shape: {sino.shape}')
print(f'Det in-bounds (both [0,511]): ', end='')
in_bounds = ((gt_2d[:,:,0] >= 0) & (gt_2d[:,:,0] <= 511) &
             (gt_2d[:,:,1] >= 0) & (gt_2d[:,:,1] <= 511))
print(f'{in_bounds.sum()}/{in_bounds.size} ({100*in_bounds.mean():.1f}%)')

print()
# Check amplitude in patch around each detected marker in shot 0
proj0 = sino[:, 0, :]
for j in range(8):
    r0, c0 = gt_2d[0, j]
    if r0 < 2 or r0 > 509 or c0 < 2 or c0 > 509:
        print(f'Marker {j}: OUT OF BOUNDS ({r0:.0f}, {c0:.0f})')
        continue
    patch = proj0[max(0,int(r0)-3):int(r0)+4, max(0,int(c0)-3):int(c0)+4]
    peak = patch.max()
    median = np.median(patch)
    amp = peak - median
    print(f'Marker {j}: GT=({r0:.1f},{c0:.1f})  peak={peak:.4f}  bg≈{median:.4f}  amp={amp:.4f}')

print()
noise_valid = noise[~np.isnan(noise)]
print(f'Noise per shot: mean={noise_valid.mean():.4f}px, max={noise_valid.max():.4f}px, '
      f'p95={np.percentile(noise_valid, 95):.4f}px')
