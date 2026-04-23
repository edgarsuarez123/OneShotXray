"""
End-to-end test: solve_shot on GT centroids must recover geometry with < 0.01 px residual.
This is the Day 3 correctness gate.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from solver.solver import solve_shot
from solver.projection import params_to_cone_vec, euler_to_R
from forward.centroiding import compute_ground_truth_projections


def _build_gt_shot(seed=42):
    rng = np.random.default_rng(seed)
    src   = np.array([0.0, 0.0, -500.0]) + rng.normal(0.0, 2.0, 3)
    det   = np.array([0.0, 0.0,  200.0]) + rng.normal(0.0, 2.0, 3)
    euler = rng.normal(0.0, 1.0, 3)
    params9_gt = np.concatenate([src, det, euler])

    det_spacing = 0.2
    det_rows = det_cols = 512
    cone_vec  = params_to_cone_vec(params9_gt, det_spacing)
    marker_3d = rng.uniform(-10.0, 10.0, (8, 3))
    gt_2d     = compute_ground_truth_projections(
        marker_3d, cone_vec[np.newaxis, :], det_rows, det_cols
    )[0]

    return dict(
        params9_gt=params9_gt,
        marker_3d=marker_3d,
        gt_2d=gt_2d,
        weights=np.full(8, 0.85),
        detection_mask=np.ones(8, dtype=bool),
        det_spacing=det_spacing,
        det_rows=det_rows,
        det_cols=det_cols,
        sod=500.0,
        odd=200.0,
    )


def test_full_shot_gt_residual():
    """
    solve_shot on GT observations must reach per_shot_rms < 0.01 px.
    This is the primary Day 3 correctness gate.
    """
    tc = _build_gt_shot()
    result = solve_shot(
        tc['marker_3d'], tc['gt_2d'], tc['weights'], tc['detection_mask'],
        tc['det_spacing'], tc['det_rows'], tc['det_cols'], tc['sod'], tc['odd'],
    )
    rms = result['per_shot_rms']
    assert rms is not None and not np.isnan(rms), 'per_shot_rms is NaN'
    assert rms < 0.01, f'End-to-end residual {rms:.6f} px > 0.01 px'


def test_position_error_gt():
    """Recovered source position must be within 0.1 mm of GT on GT observations."""
    tc = _build_gt_shot()
    result = solve_shot(
        tc['marker_3d'], tc['gt_2d'], tc['weights'], tc['detection_mask'],
        tc['det_spacing'], tc['det_rows'], tc['det_cols'], tc['sod'], tc['odd'],
    )
    pos_err = float(np.linalg.norm(result['params9'][0:3] - tc['params9_gt'][0:3]))
    assert pos_err < 0.1, f'Position error {pos_err:.4f} mm > 0.1 mm'


def test_angular_error_gt():
    """Recovered rotation must be within 0.1 deg of GT on GT observations."""
    tc = _build_gt_shot()
    result = solve_shot(
        tc['marker_3d'], tc['gt_2d'], tc['weights'], tc['detection_mask'],
        tc['det_spacing'], tc['det_rows'], tc['det_cols'], tc['sod'], tc['odd'],
    )
    R_est = euler_to_R(result['params9'][6:9])
    R_gt  = euler_to_R(tc['params9_gt'][6:9])
    R_err = R_gt.T @ R_est
    trace = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    ang_err = float(np.degrees(np.arccos(trace)))
    assert ang_err < 0.1, f'Angular error {ang_err:.4f} deg > 0.1 deg'


def test_full_shot_noisy_observations():
    """solve_shot on ~0.13px noisy observations must give per_shot_rms < 0.2 px."""
    tc = _build_gt_shot(seed=77)
    rng = np.random.default_rng(13)
    noisy_2d = tc['gt_2d'] + rng.normal(0.0, 0.13, tc['gt_2d'].shape)

    result = solve_shot(
        tc['marker_3d'], noisy_2d, tc['weights'], tc['detection_mask'],
        tc['det_spacing'], tc['det_rows'], tc['det_cols'], tc['sod'], tc['odd'],
    )
    rms = result['per_shot_rms']
    assert rms is not None and not np.isnan(rms), 'per_shot_rms is NaN'
    assert rms < 0.2, f'Noisy observations: residual {rms:.4f} px > 0.2 px gate'
