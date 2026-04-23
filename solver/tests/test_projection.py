"""
Tests for solver/projection.py

Critical: test_project_consistency_with_forward verifies that the solver's projection
exactly reproduces forward/centroiding.py::compute_ground_truth_projections.
If this fails, GT residuals won't vanish and the solver won't converge.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from solver.projection import (
    euler_to_R, R_to_euler, params_to_cone_vec,
    project_point, project_points_batch, _detector_basis,
)


def test_euler_roundtrip():
    """R_to_euler(euler_to_R(e)) must reproduce identical rotation for random angles."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = rng.uniform(-15.0, 15.0, 3)
        R1 = euler_to_R(e)
        e2 = R_to_euler(R1)
        R2 = euler_to_R(e2)
        assert np.allclose(R1, R2, atol=1e-10), f'R round-trip failed for e={e}'


def test_project_origin_to_center():
    """Origin point projects to detector center under nominal identity geometry."""
    src = np.array([0.0, 0.0, -500.0])
    det = np.array([0.0, 0.0,  200.0])
    det_spacing = 0.2
    det_rows = det_cols = 512

    u_nom, v_nom = _detector_basis(src)
    u_vec = u_nom * det_spacing
    v_vec = v_nom * det_spacing

    rc = project_point(np.zeros(3), src, det, u_vec, v_vec, det_rows, det_cols)
    center = np.array([(det_rows - 1) / 2.0, (det_cols - 1) / 2.0])
    assert np.allclose(rc, center, atol=1e-10), f'Origin should map to center, got {rc}'


def test_batch_matches_single():
    """project_points_batch must agree with project_point for all markers."""
    rng = np.random.default_rng(1)
    src = np.array([50.0, -30.0, 480.0])
    det = -src * (200.0 / 500.0)
    det_spacing = 0.2
    u_nom, v_nom = _detector_basis(src)
    u_vec = u_nom * det_spacing
    v_vec = v_nom * det_spacing

    points = rng.uniform(-10.0, 10.0, (8, 3))
    single = np.array([
        project_point(p, src, det, u_vec, v_vec, 512, 512) for p in points
    ])
    batch = project_points_batch(points, src, det, u_vec, v_vec, 512, 512)
    assert np.allclose(single, batch, atol=1e-10)


def test_params_to_cone_vec_matches_forward():
    """
    params_to_cone_vec must produce the same 12 floats as
    forward/geometry.py::geometry_to_cone_vec for the same 9-DOF params.
    """
    from forward.geometry import geometry_to_cone_vec

    params9 = np.array([50.0, -30.0, 480.0, -20.0, 12.0, -190.0, 1.2, -0.8, 0.5])

    solver_cv = params_to_cone_vec(params9, det_spacing=0.2)
    fwd_cv    = geometry_to_cone_vec(params9[np.newaxis, :], 0.2, 512, 512)[0]

    assert np.allclose(solver_cv, fwd_cv, atol=1e-12), (
        f'cone_vec mismatch:\nsolver: {solver_cv}\nforward: {fwd_cv}'
    )


def test_project_consistency_with_forward():
    """
    project_points_batch must reproduce compute_ground_truth_projections to < 1e-10 px.
    This is the critical forward/solver consistency check — any deviation causes
    non-zero GT residuals and prevents solver convergence.
    """
    from forward.centroiding import compute_ground_truth_projections

    params9 = np.array([48.3, -21.7, 495.1, -19.3, 8.7, -198.6, 0.7, -0.4, 0.3])
    det_spacing = 0.2
    det_rows = det_cols = 512

    cone_vec = params_to_cone_vec(params9, det_spacing)

    rng = np.random.default_rng(7)
    markers = rng.uniform(-10.0, 10.0, (8, 3))

    gt_2d    = compute_ground_truth_projections(
        markers, cone_vec[np.newaxis, :], det_rows, det_cols
    )[0]
    solver_2d = project_points_batch(
        markers,
        cone_vec[0:3], cone_vec[3:6],
        cone_vec[6:9], cone_vec[9:12],
        det_rows, det_cols,
    )

    max_diff = np.abs(gt_2d - solver_2d).max()
    assert max_diff < 1e-10, (
        f'forward/solver projection mismatch: max diff = {max_diff:.2e} px'
    )
