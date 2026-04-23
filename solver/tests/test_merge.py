"""
Tests for solver/merge.py — quaternion round-trip, Markley average, translation merge.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from solver.merge import euler_to_quat, quat_to_euler, quaternion_average, merge_u7_solutions
from solver.projection import euler_to_R


def test_quat_euler_roundtrip():
    """quat_to_euler(euler_to_quat(e)) must reconstruct identical rotation."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        e = rng.uniform(-15.0, 15.0, 3)
        q  = euler_to_quat(e)
        e2 = quat_to_euler(q)
        R1 = euler_to_R(e)
        R2 = euler_to_R(e2)
        assert np.allclose(R1, R2, atol=1e-10), f'Quat round-trip failed for e={e}'


def test_quaternion_average_identity():
    """Average of N identity quaternions with equal weights is identity."""
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    quats = np.tile(q_id, (8, 1))
    w = np.ones(8) / 8.0
    q_avg = quaternion_average(quats, w)
    assert np.allclose(np.abs(np.dot(q_avg, q_id)), 1.0, atol=1e-10)


def test_quaternion_average_known_rotation():
    """Average of 8 identical rotations must equal that rotation."""
    e = np.array([0.0, 5.0, 0.0])   # 5-deg pitch
    q = euler_to_quat(e)
    quats = np.tile(q, (8, 1))
    w = np.ones(8) / 8.0
    q_avg  = quaternion_average(quats, w)
    e_avg  = quat_to_euler(q_avg)
    assert np.allclose(euler_to_R(e), euler_to_R(e_avg), atol=1e-10)


def test_merge_translation_weighted_mean():
    """Merged source translation must equal the weighted mean of inputs."""
    costs = [0.05, 0.05, 0.05]
    srcs  = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    results = [
        {'params9': np.array(s + [0.0, 0.0, 200.0, 0.0, 0.0, 0.0]),
         'cost_px': c, 'success': True}
        for s, c in zip(srcs, costs)
    ]
    merged = merge_u7_solutions(results)
    # Equal costs → equal softmin weights → mean(1,2,3) = 2
    assert abs(merged[0] - 2.0) < 1e-6, f'Expected src_x=2.0, got {merged[0]:.6f}'


def test_merge_all_failed_returns_fallback():
    """When every U7 fails, return a finite fallback without NaN."""
    results = [
        {'params9': np.array([0, 0, -500, 0, 0, 200, 0, 0, 0], dtype=float),
         'cost_px': 5.0, 'success': False}
        for _ in range(8)
    ]
    merged = merge_u7_solutions(results)
    assert not np.any(np.isnan(merged)), 'Fallback must not contain NaN'
    assert np.all(np.isfinite(merged)), 'Fallback must be finite'
