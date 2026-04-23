"""
forward/ — SDSG forward projection + centroiding package.

Sinogram axis order: (det_rows, N_shots, det_cols)  ← document everywhere, common bug source
ASTRA volume axis order: (Z, Y, X) — transpose from phantom's (X, Y, Z) before passing to ASTRA
All coordinates in mm unless noted.
"""
from .geometry import generate_hemisphere_shots, perturb_geometry, geometry_to_cone_vec
from .projector import forward_project, apply_noise_pipeline
from .centroiding import (
    detect_markers,
    match_identities,
    refine_centroids,
    compute_ground_truth_projections,
    measure_centroiding_noise,
    compute_geometric_unsharpness,
    compute_unsharpness_weights,
    process_all_shots,
)
