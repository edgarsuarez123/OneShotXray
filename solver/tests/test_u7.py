"""
Tests for solver/u7.py

test_residual_at_gt_is_zero is the CRITICAL gate test: if GT params produce
non-zero residuals the coordinate convention is broken and nothing will work.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from solver.projection import params_to_cone_vec
from solver.u7 import u7_residuals, solve_u7
from forward.centroiding import compute_ground_truth_projections


def _gt_test_case(seed=42):
    """Build a deterministic GT test case with 8 markers."""
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
    weights = np.full(8, 0.85)

    return dict(
        params9_gt=params9_gt,
        marker_3d=marker_3d,
        gt_2d=gt_2d,
        weights=weights,
        det_spacing=det_spacing,
        det_rows=det_rows,
        det_cols=det_cols,
        sod=500.0,
        odd=200.0,
    )


def test_residual_at_gt_is_zero():
    """
    CRITICAL: u7_residuals at GT params must be < 1e-6 px for all anchors.
    Failure here means forward/solver coordinate mismatch — fix before proceeding.
    """
    tc = _gt_test_case()
    for anchor in range(8):
        resid = u7_residuals(
            tc['params9_gt'],
            tc['marker_3d'],
            tc['gt_2d'],
            tc['weights'],
            anchor_idx=anchor,
            det_spacing=tc['det_spacing'],
            det_rows=tc['det_rows'],
            det_cols=tc['det_cols'],
        )
        max_r = float(np.max(np.abs(resid)))
        assert max_r < 1e-6, (
            f'GT residual {max_r:.2e} px at anchor={anchor} — convention mismatch'
        )


def test_residual_magnitude_with_noise():
    """Residuals on noisy observations (~0.13 px) should be in a sane range."""
    tc = _gt_test_case()
    rng = np.random.default_rng(99)
    noisy_2d = tc['gt_2d'] + rng.normal(0.0, 0.13, tc['gt_2d'].shape)

    resid = u7_residuals(
        tc['params9_gt'],
        tc['marker_3d'],
        noisy_2d,
        tc['weights'],
        anchor_idx=0,
        det_spacing=tc['det_spacing'],
        det_rows=tc['det_rows'],
        det_cols=tc['det_cols'],
    )
    # Rough sanity: weighted RMS / median_weight gives approx pixel noise
    med_w = float(np.median(tc['weights']))
    rms_px = float(np.sqrt(np.mean(resid**2))) / med_w
    assert 0.01 < rms_px < 2.0, (
        f'Noisy residual magnitude {rms_px:.4f} px outside expected range'
    )


def test_single_u7_converges_on_gt_observations():
    """solve_u7 on GT observations must converge to cost < 0.01 px."""
    tc = _gt_test_case()
    result = solve_u7(
        tc['marker_3d'],
        tc['gt_2d'],
        tc['weights'],
        anchor_idx=3,
        det_spacing=tc['det_spacing'],
        det_rows=tc['det_rows'],
        det_cols=tc['det_cols'],
        sod=tc['sod'],
        odd=tc['odd'],
    )
    assert result['cost_px'] < 0.01, (
        f'solve_u7 on GT obs: cost_px={result["cost_px"]:.6f} > 0.01 px'
    )
    assert result['nfev'] <= 200, f'Exceeded max_nfev: {result["nfev"]}'


def test_single_u7_converges_nominal_perturbation():
    """solve_u7 on noisy observations (0.13 px centroid noise) must give cost < 0.2 px."""
    tc = _gt_test_case()
    rng = np.random.default_rng(7)
    noisy_2d = tc['gt_2d'] + rng.normal(0.0, 0.13, tc['gt_2d'].shape)

    result = solve_u7(
        tc['marker_3d'],
        noisy_2d,
        tc['weights'],
        anchor_idx=0,
        det_spacing=tc['det_spacing'],
        det_rows=tc['det_rows'],
        det_cols=tc['det_cols'],
        sod=tc['sod'],
        odd=tc['odd'],
    )
    assert result['cost_px'] < 0.2, (
        f'solve_u7 on noisy obs: cost_px={result["cost_px"]:.4f} > 0.2 px gate'
    )
