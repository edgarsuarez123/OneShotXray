---
title: "SDSG SBIR Preliminary Data PRD"
subtitle: "Navy DON26BZ01-NV012 | NIH NIBIB Bedside Brain CT"
organization: "Darrow Industries, Inc."
lead_engineer: "Edgar J. Suarez Colon"
date: "April 21, 2026"
deadline: "April 29, 2026"
version: "1.0"
classification: "CONFIDENTIAL"
---

# PRODUCT REQUIREMENTS DOCUMENT
## SDSG SBIR Preliminary Data Generation
**Navy DON26BZ01-NV012 | NIH NIBIB Bedside Brain CT**
Darrow Industries, Inc. | Lead Engineer: Edgar J. Suarez Colon
April 21, 2026 | Submission Deadline: April 29, 2026 | Version 1.0 | **CONFIDENTIAL**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Technical Background and Patent Fidelity Requirements](#2-technical-background-and-patent-fidelity-requirements)
3. [Shared Infrastructure Requirements](#3-shared-infrastructure-requirements-both-sbirs)
4. [SBIR 1 — Navy DON26BZ01-NV012](#4-sbir-1--navy-don26bz01-nv012-specific-requirements)
5. [SBIR 2 — NIH NIBIB Bedside Brain CT](#5-sbir-2--nih-nibib-bedside-brain-ct-specific-requirements)
6. [Five-Day Execution Plan](#6-five-day-execution-plan--april-21-25-2026)
7. [Risk Register and Fallback Protocols](#7-risk-register-and-fallback-protocols)
8. [Required Citations — Path 1 Literature Foundation](#8-required-citations--path-1-literature-foundation)
9. [Definition of Done — Acceptance Checklist](#9-definition-of-done--acceptance-checklist)
10. [Roles and Responsibilities](#10-roles-and-responsibilities)

---

## 1. Executive Summary

This PRD defines all requirements, deliverables, acceptance criteria, and a day-by-day execution plan for generating computational preliminary data for two concurrent SBIR Phase I proposals submitted by Darrow Industries, Inc. Both proposals are grounded in U.S. Patent 12,327,378 B2 — the Self-Determined Shot Geometry (SDSG) system invented by Case and Kenderian of The Aerospace Corporation.

The same core algorithm is applied to two distinct markets: naval ship hull crack detection (Navy SBIR, due April 29) and ICU bedside brain hemorrhage monitoring (NIH SBIR, due April 29). Preliminary data must demonstrate algorithmic feasibility at the computational simulation level. SBIR Phase I reviewers do not expect physical hardware — they expect proof that the math works under realistic conditions, with clearly stated metrics and citations anchoring results to peer-reviewed prior work.

| Item | Navy SBIR | NIH SBIR |
|---|---|---|
| Proposal Topic | DON26BZ01-NV012 | NIBIB Phase I SBIR |
| Full Title | SDSG for 3D Characterization of Ship Hull Fatigue and Corrosion | Bedside Brain Injury Monitoring via Open-Configuration Portable CT |
| Hard Deadline | April 29, 2026 | April 29, 2026 |
| Phase I Budget | ~$240,000 | ~$300,000 |
| Phantom Material | Steel block, 25mm thick | Ellipsoidal skull with brain tissue interior |
| Target Defect | 0.8mm fatigue crack (air void) | 5mm intracranial hemorrhage (DeltaHU >= 40) |
| Geometry Accuracy Target | < 0.2 pixels residual error | < 0.3 pixels residual error |
| Shot Arc Constraint | Full 360-degree hemisphere | Restricted 180-degree arc (ICU bedside) |
| Primary Reconstruction | mART vs FBP | mART vs FBP vs SART |
| Primary Pass Metric | CNR >= 4 at 0.8mm crack (Rose criterion) | Simulated AUC > 0.75 for 5mm hemorrhage |
| Required Citations | 5 (shared + Navy-specific) | 9 (shared + NIH-specific) |

---

## 2. Technical Background and Patent Fidelity Requirements

All simulation work must be architecturally faithful to U.S. Patent 12,327,378 B2 and the primary publication: Case, J.T. and Kenderian, S., *"Self-Determined Shot Geometry for Open-Configuration Portable X-Ray CT,"* IEEE Open Journal of Instrumentation and Measurement, 2023, DOI: 10.1109/OJIM.2023.3268451. No simplifications that violate the patent's core claims are permitted.

### 2.1 The Core Problem

Conventional CT uses a rigid gantry so source-detector geometry is always known. In an open-configuration portable system, the technician moves source and detector freehand with no mechanical link. For each shot, the system does not know where it was. This is the "geometric uncertainty" problem the patent solves.

### 2.2 The 9-Unknown Geometry Problem

Each X-ray shot has 9 geometric unknowns: 3 source position coordinates (X, Y, Z), 3 detector center coordinates (X, Y, Z), and 3 detector orientation angles (Euler rotations). Solving all 9 simultaneously from noisy 2D marker positions is a nonlinear optimization problem prone to local minima. The patent solves this with the 07/09 split.

### 2.3 The 07/09 Optimization Split — Core Patent Claim

For each shot, one fiducial marker is chosen as the "anchor." The coordinate system is translated so that marker is at the origin, which removes 2 degrees of freedom, reducing the problem from 9 unknowns to 7 (called a U7 problem). This is solved once per marker (6–8 U7 problems per shot). All U7 solutions are evaluated by their final cost and merged using fitness-weighted averaging. The rotation component is averaged using quaternion math to prevent gimbal lock.

### 2.4 mART Reconstruction

After recovering geometry for all shots, the 3D volume is reconstructed using the Multiplicative Algebraic Reconstruction Technique (mART). Unlike FBP which requires full circular orbit data, mART is an iterative solver that works with arbitrary shot trajectories and sparse views. It uses a multiplicative update rule and converges faster than additive ART for non-negative attenuation data. GPU acceleration via Siddon ray-tracing (ASTRA Toolbox) makes 50-iteration runs feasible in under 30 seconds.

---

## 3. Shared Infrastructure Requirements (Both SBIRs)

The following requirements apply to both simulations. The Navy and NIH workstreams share the same computational environment, forward projector, centroiding module, SDSG solver, and reconstruction engine. Only the phantom geometry and shot configuration differ.

### 3.1 Development Environment

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| ENV-001 | Python 3.11 via Miniforge or Miniconda conda environment named `sdsg_sim` | `python --version` returns 3.11.x in activated environment |
| ENV-002 | ASTRA Toolbox 2.4.1 installed via conda from astra-toolbox channel with CUDA support | `astra.test()` executes and reports zero failures on all CUDA tests |
| ENV-003 | NVIDIA GPU with CUDA 11+ and minimum 8GB VRAM available. Fallback: TIGRE package CPU mode if no GPU. | `nvidia-smi` returns valid device. Or TIGRE imports cleanly if GPU unavailable. |
| ENV-004 | NumPy >= 1.26, SciPy >= 1.13, scikit-image >= 0.22, matplotlib >= 3.8, h5py >= 3.10, tqdm >= 4.66 all installed | All packages import without error. `skimage.metrics.structural_similarity` callable. |
| ENV-005 | Git repository initialized. Folder structure: `phantom/`, `forward/`, `solver/`, `recon/`, `analysis/`, `figures/`, `data/navy/`, `data/nih/`, `notebooks/` | `git log` shows initial commit. All folders present. `.gitignore` excludes large HDF5 data files. |
| ENV-006 | `requirements.txt` or `environment.yml` committed with pinned versions | Clean environment recreation from file produces passing `astra.test()` |

### 3.2 Forward Projection Engine

The forward projector simulates the physical X-ray measurement process, producing 2D projection images from a 3D volume under a given shot geometry. This is the "digital twin" of the physical device.

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| FWD-001 | Use ASTRA `cone_vec` geometry type exclusively. Mandatory — `cone_vec` supports fully arbitrary per-shot geometries unlike the standard `cone` geometry which requires circular orbit. | All ASTRA geometry objects use type `cone_vec`. Verified in code review. |
| FWD-002 | Detector: 512 x 512 pixels, 0.2mm pixel pitch. Represents industrial/medical flat-panel detector class. | ASTRA detector configured with `det_row_count=512`, `det_col_count=512`, `det_spacing_x=0.2`, `det_spacing_y=0.2` |
| FWD-003 | Nominal scan parameters: Source-to-Object Distance (SOD) = 500mm, Object-to-Detector Distance (ODD) = 200mm. | Nominal V matrix rows computed from these distances. Documented in README. |
| FWD-004 | Per-shot geometry perturbation simulating open-configuration uncertainty. Source position: Gaussian noise sigma_s = 2mm (nominal), 10mm (stress). Detector Euler angles: Gaussian noise sigma_theta = 1 degree (nominal), 5 degrees (stress). Ground truth 9-DOF stored per shot. | Ground truth geometry HDF5 array shape (N_shots, 9). Random seed fixed at 42. |
| FWD-005 | Beer-Lambert conversion: convert to intensity via `I = I0 * exp(-sino)` with I0 = 10,000 photons/pixel (typical industrial radiograph SNR). | Intensity images have values in range [0, I0]. Not negative. |
| FWD-006 | Poisson photon noise applied to intensity image before log conversion back to attenuation. | Noise visible in projections. SNR approximately 100:1 at I0=10000. |
| FWD-007 | Geometric unsharpness modeled as Gaussian blur per projection. Sigma = focal_spot_size * ODD / SOD / 2.355. Focal spot size = 0.5mm, resulting in sigma = 0.2mm = 1 detector pixel. | Blur applied per projection slice. Sigma value confirmed = 0.2mm. |
| FWD-008 | ASTRA sinogram axis order verified and documented: `(det_rows, N_projections, det_cols)`. This is a common bug source causing incorrect 3D indexing. | Axis order documented in code comments. Single test shot verified manually. |

### 3.3 Fiducial Marker Centroiding Module

For each 2D projection image, the centroiding module must locate the exact sub-pixel positions of all fiducial markers. This is a critical accuracy bottleneck — centroiding error propagates directly into reprojection error and must be less than 0.15 pixels RMS to support the overall geometric accuracy targets.

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| CENT-001 | Lead/BaSO4 markers project as dark disks (high attenuation). Invert projection image before blob detection to convert dark disks to bright peaks. | Inverted image shows bright blobs at all marker locations. |
| CENT-002 | Coarse detection: `skimage.feature.blob_log` with `min_sigma=2`, `max_sigma=8`, `num_sigma=12`, `threshold=0.05`, `exclude_border=True`. | Correct number of blobs detected in >= 95% of projections. Logged per shot. |
| CENT-003 | Marker identity assignment: match detected blobs to known marker identities using nearest-neighbor matching against initial-guess reprojection from nominal geometry. | Zero identity swap errors across all shots. Confirmed by consistency check across adjacent shots. |
| CENT-004 | Sub-pixel centroid refinement: `scipy.ndimage.center_of_mass` applied within 20-pixel window around each coarse centroid. Refined centroid stored in HDF5. | Centroid positions consistent across repeated runs. Stored as float64, not rounded to integer. |
| CENT-005 | Centroiding noise measurement: for each shot, compute RMS pixel difference between refined centroid and known ground-truth projection. Target: < 0.15px RMS. | Centroiding noise stored in HDF5 key `centroids/noise_per_shot`. Aggregate mean < 0.15px confirmed. |
| CENT-006 | Unsharpness confidence weight: `w = 1 / (1 + U_g / pixel_size)` where U_g is geometric unsharpness. Pass as weight vector to cost function. | Weight vector of length N_markers produced per shot. Values between 0 and 1. |

### 3.4 SDSG Solver — The 07/09 Optimization Split

This is the patent's primary contribution. Implementation must be exact — no approximations. Any deviation from the patent's algorithm description is a requirement violation.

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| SDSG-001 | For each shot, execute one U7 sub-problem per marker. If N_markers = 6, execute 6 U7 problems. If N_markers = 8, execute 8. This is the "anchor iteration" loop. | N U7 solutions produced per shot where N = N_markers. Verified in unit test. |
| SDSG-002 | U7 coordinate translation: subtract anchor marker 3D position from all other marker positions. Anchor becomes the origin. Reduces 9 unknowns to 7 by fixing the anchor's projection relationship. | Translated coordinate frame verified: anchor position = [0,0,0] in translated frame. |
| SDSG-003 | 7-unknown parameter vector per U7 problem: `[source_x, source_y, source_z, detector_z_along_anchor_ray, Euler_alpha, Euler_beta, Euler_gamma]`. Detector x and y derived analytically from the constraint that anchor must project to its measured centroid pixel. | Parameter vector has exactly 7 elements. Detector x,y derivation confirmed analytically. |
| SDSG-004 | Cost function: for given x7, construct full pinhole projection model. Project all non-anchor markers through source onto detector plane. Compute pixel-space distance to measured centroid. Return flattened residual vector of length `2*(N_markers-1)` in pixel units. | Returns near-zero residuals when fed ground-truth geometry. Units confirmed as pixels. |
| SDSG-005 | Optimizer: `scipy.optimize.least_squares`, `method='lm'` (Levenberg-Marquardt). Hyperparameters: `xtol=1e-10`, `ftol=1e-10`, `max_nfev=200`, `x_scale='jac'`. | Optimizer configured exactly as specified. No other optimizer substitutions without documenting as a fallback. |
| SDSG-006 | Initial guess: source at `[0, 0, -SOD]`, detector at `[0, 0, ODD]`, identity rotation `(Euler = [0,0,0])`. Same initial guess for all anchors within a shot. | Solver converges from this guess under nominal perturbation (sigma_s=2mm, sigma_theta=1deg) in < 100 function evaluations. |
| SDSG-007 | Fitness-weighted merge: compute softmin weights `w_k = exp(-J_k / J_min) / sum(exp(-J_k / J_min))`. Weighted average of 9-DOF solutions gives merged estimate. **Rotation merged using quaternion weighted average** — not naive Euler averaging (causes gimbal lock). | Weights sum to 1.0. Quaternion averaging implemented and confirmed. |
| SDSG-008 | Residual measurement: after merge, compute final per-marker reprojection error vs measured centroids. Store per-shot mean, overall mean, median, 95th percentile in HDF5 under `results/residuals/`. | All residual statistics present in HDF5. Overall mean < 0.2px (Navy) or < 0.3px (NIH) under nominal conditions. |
| SDSG-009 | Failed U7 handling: if any U7 final cost > 2.0px after max_nfev, flag as failed. Log warning with shot index and anchor index. Exclude from merge. Pipeline continues. | Failed U7 count reported. No crashes on failure. |
| SDSG-010 | Stress test: run full pipeline with sigma_s=10mm, sigma_theta=5deg. Store separately under `results/stress_test/`. Expected residuals 0.3–0.8px range. | Stress test results present in HDF5. Pipeline completes without crashing. |

### 3.5 Volumetric Reconstruction Module

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| RECON-001 | FBP baseline: ASTRA `FDK_CUDA` algorithm applied to recovered `cone_vec` geometry. Label in all figures as **"FBP (baseline, approximate for non-circular trajectory)"** — mandatory label to be honest with reviewers. | FBP volume produced. Correct label used in all proposal figures. |
| RECON-002 | mART: custom Python loop using ASTRA forward/backward projectors as primitives. Multiplicative update rule: `x_new = x * exp(A_T * log(b / Ax) / col_sum)`. All intermediate values clamped at epsilon = 1e-6 minimum. | mART volume produced in 50 iterations. No NaN, no Inf, no negative values. |
| RECON-003 | mART pre-computation: column sum vector (`A_T` applied to sinogram of all-ones) computed once before the iteration loop. Not recomputed each iteration. | Single backprojection of ones at initialization. `col_sum` shape matches volume shape. |
| RECON-004 | mART iterations: 50 for production runs. Per-iteration relative update norm stored as convergence diagnostic. | Convergence array of length 50 stored per run. Values monotonically decreasing. |
| RECON-005 | Memory management: all ASTRA `data3d` and algorithm objects explicitly deleted after each reconstruction to prevent CUDA memory leak. | Memory footprint stable across 20 consecutive trials. No CUDA out-of-memory errors. |
| RECON-006 | Fiducial marker inpainting: voxels within 1.5mm radius of each known marker 3D position replaced with mean attenuation of surrounding non-marker voxels (3mm annular shell). | Marker artifacts absent in final reconstruction slices used in proposal figures. |
| RECON-007 | Run time logging: wall-clock time per complete pipeline run recorded in HDF5 metadata and printed to console. | Logged time < 8 minutes on RTX 3060-class GPU for 100-shot Navy run. |

### 3.6 Image Quality Metrics Module

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| METRIC-001 | SSIM: `skimage.metrics.structural_similarity` with `data_range = phantom.max() - phantom.min()` explicitly set. `gaussian_weights=True`, `sigma=1.5`, `use_sample_covariance=False`. **CRITICAL: `data_range` must be passed explicitly — the default halves SSIM silently for float arrays.** | SSIM value between 0 and 1. Verified `data_range` parameter is passed. |
| METRIC-002 | PSNR: `skimage.metrics.peak_signal_noise_ratio` with `data_range` explicitly set. Same note applies. | PSNR value in dB. Verified `data_range` parameter is passed. |
| METRIC-003 | CNR at defect: `CNR = |mean(recon[defect_mask]) - mean(recon[bg_mask])| / std(recon[bg_mask])`. Defect mask = 3-voxel dilation of known defect geometry. Background mask = matched-volume ROI in homogeneous undamaged region. | CNR computed for each defect size and reconstruction method. Stored in HDF5 under `metrics/{method}/cnr_{defect_id}`. |
| METRIC-004 | All metrics computed for each reconstruction method (FBP, mART, and SART for NIH only). Organized in HDF5 under `metrics/{method_name}/`. | All method/metric combinations present in HDF5. |
| METRIC-005 | Position recovery error: Euclidean distance between recovered and ground-truth source position, in mm, per shot. Angular recovery error: geodesic rotation distance, in degrees, per shot. | Arrays `position_error_mm` and `angular_error_deg` of length N_shots stored in `results/geometry_accuracy/`. |

---

## 4. SBIR 1 — Navy DON26BZ01-NV012: Specific Requirements

The Navy SBIR must demonstrate that SDSG can recover accurate shot geometry and reconstruct internal fatigue cracks in thick steel. Every Phase I Technical Objective in Volume 2 of the proposal must be directly addressed by a specific figure or table. No objective may be left without supporting data.

| Proposal Objective | Metric | Target | Preliminary Data Required |
|---|---|---|---|
| Obj 1: Sub-Voxel Geometric Accuracy | Mean reprojection residual (pixels) | < 0.2 px | NV-FIG-01: Residual histogram |
| Obj 2: 07/09 Splitting Robustness | Convergence success rate vs. non-split baseline | Split converges reliably, baseline stalls | NV-FIG-03 annotation + NV-TAB-01 |
| Obj 3: mART vs FBP Fidelity | SSIM, PSNR, CNR comparison | mART SSIM > FBP SSIM at all crack widths | NV-FIG-02 + NV-TAB-02 |
| Obj 4: 0.8mm Crack Resolution in 25mm Steel | CNR at 0.8mm crack per ASTM E1742 | CNR >= 4 (Rose criterion) | NV-FIG-04 + NV-TAB-02 |

### 4.1 Navy Steel Phantom Requirements

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| NV-PHN-001 | Phantom geometry: 25mm x 25mm x 25mm steel block. Voxel size: 0.1mm isotropic. Grid: 250 x 250 x 250. | HDF5 volume shape `(250,250,250)`. Metadata records `voxel_size_mm = 0.1`. |
| NV-PHN-002 | Steel attenuation: `mu_steel = 0.1149 mm^-1` (NIST XCOM, iron at 200keV). Energy chosen because at 100keV, 25mm steel transmits only 0.065% (unusable SNR); at 200keV, transmission is 5.7% (adequate). | All steel voxels = 0.1149 float32. Energy choice documented in proposal methods. |
| NV-PHN-003 | Primary crack (proposal target): planar air void 0.8mm wide, full length of phantom, centered at 12mm depth. This is the ASTM E1742 compliance target. | Crack width = 0.8mm = 8 voxels at 0.1mm pitch. Visible in cross-section. |
| NV-PHN-004 | Secondary crack (sanity check): 1.6mm wide crack at same depth. Expected to be clearly visible. | 1.6mm crack present and visible. |
| NV-PHN-005 | Stretch-target crack (below spec): 0.4mm wide crack. Expected to fail Rose criterion. Demonstrates system resolution limit honestly. | 0.4mm crack present. Proposal labels this "below Phase I specification, targeted in Phase II." |
| NV-PHN-006 | All crack voxels use sub-voxel partial volume weighting: `attenuation = mu_steel * (1 - fractional_intersection_with_crack_plane)`. Prevents staircase artifacts at 0.1mm resolution. | Crack edges smooth in cross-section. No binary-mask staircase pattern. |
| NV-PHN-007 | Lead fiducial constellation: 6–8 spheres, 2mm diameter. `mu_lead = 1.133 mm^-1` at 200keV. Non-coplanar (no 4 markers share a plane) and non-symmetric. | Marker matrix (N,3) stored in HDF5. Non-coplanarity confirmed: determinant of any 4-marker sub-matrix non-zero. |
| NV-PHN-008 | Two gas porosity inclusions: spherical voids 0.5mm and 1.0mm diameter, mu=0. Demonstrates multi-defect detection capability. | Two void spheres present in ground-truth volume. |
| NV-PHN-009 | Saved to `data/navy/phantom.h5` with datasets: `volume` (float32 250x250x250), `marker_positions` (float32 N×3), `voxel_size_mm`, `crack_widths_mm`, `crack_depths_mm`. | HDF5 file present. All datasets readable with h5py. |

### 4.2 Navy Shot Configuration Requirements

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| NV-SHOT-001 | Primary run: N = 100 shots. Additional runs at N = 20, 50, 200 for accuracy vs shot count figure. | Four separate HDF5 sinogram files for N = 20, 50, 100, 200. |
| NV-SHOT-002 | Shot distribution: full 360-degree hemisphere above phantom. Positions sampled uniformly (Fibonacci sphere or random uniform on unit hemisphere), scaled to SOD = 500mm. | No clustering. Coverage spans full azimuth range 0–360 degrees. |
| NV-SHOT-003 | Both nominal (sigma_s=2mm, sigma_theta=1deg) and stress test (sigma_s=10mm, sigma_theta=5deg) perturbations stored in same HDF5 under `shots/nominal/` and `shots/stress/`. | Both perturbation groups present in HDF5. |
| NV-SHOT-004 | Random seed = 42 fixed for all Navy runs. Any re-run by any person must produce identical figures. | Confirmed: running script twice with seed=42 produces bit-identical HDF5 files. |

### 4.3 Navy Deliverables — Figures and Tables

> **Figure spec:** All figures saved at minimum 300 DPI, 8×6 inches, font size >= 12pt. Figures will be directly embedded in the SBIR proposal document.

| Deliverable ID | Title | Content Description | Pass/Fail Criterion |
|---|---|---|---|
| **NV-FIG-01** | SDSG Geometric Accuracy: Reprojection Residual Distribution | Histogram of per-marker reprojection residuals (px) across all shots at N=100. X-axis: 0–1.0px. Y-axis: count. Vertical dashed red line at 0.2px target. Text annotation: mean and 95th percentile. Caption: "Figure 1. Distribution of per-marker reprojection residuals across 100 simulated shots under nominal open-configuration perturbation (sigma_s=2mm, sigma_theta=1 degree). Mean residual [X] px. 95th percentile [X] px. Target: < 0.2 px." | Mean residual < 0.2px. 95th percentile < 0.5px. Saved as `figures/navy/fig01_residual_histogram.png`. |
| **NV-FIG-02** | SDSG Reconstruction: mART vs FBP at 0.8mm Crack | Three-panel figure: (A) Ground Truth, (B) FBP reconstruction, (C) mART reconstruction. Same axial slice through 0.8mm crack. Identical grayscale window/level. 2mm scale bar. White arrow to crack in each panel. Subtitles: "A: Ground Truth", "B: FBP (baseline)", "C: mART". Caption with image quality metrics. | Crack visually distinct in panel C. Blurred/absent in panel B. CNR values in caption. |
| **NV-FIG-03** | SDSG Geometric Accuracy vs. Number of X-Ray Shots | Line plot. X-axis: number of shots (20, 50, 100, 200). Y-axis: mean residual (px). Error bars = 1 std. Dashed horizontal line at 0.2px. Caption: more shots improves accuracy via overdetermination of geometry. | All four data points present. Curve shows decreasing trend. |
| **NV-FIG-04** | CNR vs. Crack Width: mART vs FBP | Line plot. X-axis: crack width mm (0.4, 0.8, 1.6). Y-axis: CNR. Two lines: FBP (dashed blue), mART (solid red). Dashed green line at CNR=4 labeled "Rose Criterion (minimum detectable)." ASTM E1742 reference in caption. | mART line >= 4 at 0.8mm and 1.6mm. FBP line < 4 at 0.8mm. |
| **NV-TAB-01** | Table 1: Shot Geometry Recovery Accuracy | Columns: Perturbation Condition, Mean Position Error (mm), Std Position Error (mm), Mean Angular Error (deg), Mean Reprojection Residual (px), 95th Pct Residual (px). Two rows: Nominal and Stress Test. | All cells populated. Nominal residual < 0.2px. |
| **NV-TAB-02** | Table 2: Reconstruction Image Quality Summary | Columns: Method, SSIM, PSNR (dB), CNR@0.4mm, CNR@0.8mm, CNR@1.6mm. Two rows: FBP and mART. | All cells populated. mART SSIM > FBP SSIM. mART CNR@0.8mm >= 4. |

### 4.4 Navy Proposal Language Requirements

The following statements must appear verbatim or substantively in the Navy proposal's preliminary data section. These address specific reviewer concerns identified from the SBIR Volume 2 text.

> **Framing statement (required):** *"Preliminary data presented in this section constitutes a computational feasibility demonstration using ray-traced simulated cone-beam X-ray data with patent-faithful geometric perturbation modeling (source position uncertainty sigma_s = 2mm, detector orientation uncertainty sigma_theta = 1 degree, representing realistic open-configuration freehand scanning conditions)."*

> **Prior real-data validation (required):** *"The foundational SDSG algorithm has been validated by its inventors on real radiographic data. Case and Kenderian (IEEE Open Journal of Instrumentation and Measurement, 2023, DOI: 10.1109/OJIM.2023.3268451) demonstrate that SDSG achieves satisfactorily low geometric error for successful CT reconstruction in non-destructive evaluation settings. The present computational feasibility demonstration independently replicates this result under the specific geometric and material conditions of naval hull inspection."*

> **Accuracy contextualization (required, references Blumensath 2024):** *"Comparable fiducial-based calibration methods published in the peer-reviewed literature achieve 0.18–0.27 pixel residual geometric error (Blumensath et al., Sensors, 2024), confirming our Phase I target of < 0.2 pixels is at the current demonstrated state of the art — ambitious but credible."*

> **Standards compliance statement (required):** *"Success is evaluated against ASTM E1742/E1742M-23, the governing standard for radiographic examination of metallic materials. The 2-2T quality level corresponds to 2% sensitivity at the material thickness — equivalent to a 0.5mm hole through 25mm steel. Our 0.8mm crack resolution target provides a 60% margin above this standard."*

> **Phase II bridge (required):** *"Phase II will replace simulated sinograms with real radiographic measurements on ASTM E1742/E1025-compliant hole-type IQI ladder test specimens machined from 10mm, 25mm, and 50mm steel plate, operated by NAS 410 Level II/III-certified radiographic testing personnel."*

---

## 5. SBIR 2 — NIH NIBIB Bedside Brain CT: Specific Requirements

The NIH SBIR must demonstrate that SDSG — when adapted for cranial geometry and constrained to ICU bedside access angles — achieves the geometric accuracy and image contrast needed to detect simulated intracranial hemorrhage.

| Specific Aim | What It Requires | Preliminary Data Required | What Awaits Phase I |
|---|---|---|---|
| Aim 1: SDSG optimized for cranial anatomy | Simulation across constellation geometry parameter grid. mART parameter sweep. Geometric accuracy < 0.3px under restricted 180-degree arc. | NIH-AIM1-01 through NIH-AIM1-05: constellation heatmap, convergence curves, geometry accuracy report | Physical calibration phantom fabrication and validation |
| Aim 2: Hemorrhage detectability in phantom | 50 scan acquisitions on CIRS 603A anthropomorphic head phantom. MRMC-ROC blinded reader study. AUC >= 0.85. | NIH-AIM2-01 through NIH-AIM2-05: slice comparison figures, CNR curves, simulated ROC estimate | CIRS 603A phantom procurement, physical scans, 5-reader blinded study |
| Aim 3: ICU operational feasibility | EMI characterization, radiation dosimetry, clinician interviews, FDA regulatory pathway draft | **None required — this is Phase I work by definition** | All Aim 3 work |

### 5.1 NIH Cranial Phantom Requirements

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| NIH-PHN-001 | Outer skull geometry: prolate ellipsoid with semi-axes 90mm × 70mm × 65mm (L–R × A–P × S–I). Approximates adult head dimensions. | Ellipsoid boundary verified. Outer semi-axes match specification. |
| NIH-PHN-002 | Skull wall: cortical bone shell 7mm thick (inner semi-axes = outer minus 7mm each). Voxel size 1.0mm isotropic. Grid approximately 200 × 160 × 140. | Shell thickness = 7mm in cross-section. Grid dimensions within 10% of specification. |
| NIH-PHN-003 | Skull attenuation: `mu_bone = 0.048 mm^-1` (NIST XCOM, cortical bone at 70keV — appropriate for head CT kVp range). | Skull voxels = 0.048 float32. Energy choice documented. |
| NIH-PHN-004 | Brain tissue interior: `mu_brain = 0.021 mm^-1` (soft tissue at 70keV, approximately 40 HU). Fills all voxels inside inner ellipsoid. | Brain interior voxels = 0.021 float32. |
| NIH-PHN-005 | Primary hemorrhage insert: sphere 5mm diameter at 25mm depth from inner skull surface. `mu_blood = 0.023 mm^-1` (fresh blood at 70keV, DeltaHU = 40–50). Primary detection target per Aim 2 Go/No-Go criteria. | 5mm sphere present. `mu_blood = 0.023`. DeltaHU >= 40 confirmed. |
| NIH-PHN-006 | Additional lesion sizes: 3mm, 8mm, 12mm spheres at same depth. The 3mm lesion is the minimum actionable threshold per neurosurgical standards. | Four lesion sizes present. Stored as separate volume files or non-overlapping inserts. |
| NIH-PHN-007 | Cerebral edema region: low-attenuation hemisphere 15mm radius, `mu = 0.019 mm^-1` (approximately −20 HU), placed contralateral to hemorrhage. | Edema region present. mu = 0.019. Placed contralateral to hemorrhage. |
| NIH-PHN-008 | Fiducial constellation: 8 markers, 2mm diameter. `mu_BaSO4 = 0.31 mm^-1` at 70keV. Anatomical landmark positions: bilateral mastoid eminences (2), bilateral temporal (2), vertex (1), bilateral parietal (2), forehead (1). | 8 markers present. Non-coplanar. Positions documented in HDF5 with anatomical labels. |
| NIH-PHN-009 | No-lesion variant: identical phantom with all hemorrhage inserts replaced by brain tissue (mu = 0.021). Required for sensitivity/specificity calculation. | Separate HDF5 file. Verified identical except at insert voxels. |
| NIH-PHN-010 | Phantom files: `data/nih/phantom_lesion_5mm.h5` (primary), `data/nih/phantom_nolesion.h5`, `data/nih/phantom_lesion_3mm.h5`, `data/nih/phantom_lesion_8mm.h5`, `data/nih/phantom_lesion_12mm.h5`. | All five HDF5 files present and readable. |

### 5.2 NIH Shot Configuration Requirements

| Req ID | Requirement | Acceptance Criterion |
|---|---|---|
| NIH-SHOT-001 | **PRIMARY:** Restricted 180-degree lateral arc. Source positions constrained to azimuth 0–180 degrees (left hemisphere only, simulating ICU bedside access from one side). No shots from inferior, posterior, or contralateral hemispheres. | All source position azimuths in [0, 180] degree range. Verified with 3D scatter plot. |
| NIH-SHOT-002 | **COMPARISON:** Full 360-degree arc. Run for comparison to quantify performance degradation from restricted access. | Two shot geometry sets: `shots/restricted_180/` and `shots/full_360/` in HDF5. |
| NIH-SHOT-003 | Number of shots: N = 80 for primary run (fewer than Navy because ICU scan time must be < 15 minutes per Aim 2 protocol). | N_shots = 80 in restricted arc dataset. N_shots = 80 in full arc dataset. |
| NIH-SHOT-004 | Perturbation: `sigma_s = 3mm` (larger than Navy due to less controlled freehand motion around patient head), `sigma_theta = 1.5 degrees`. | Perturbation values documented in HDF5 metadata and README. |
| NIH-SHOT-005 | All attenuation coefficients use 70keV NIST values throughout. **Do not mix 200keV (Navy) and 70keV (NIH) values in the same simulation.** | Energy level documented in HDF5 metadata. No cross-contamination with Navy energy values. |
| NIH-SHOT-006 | SDSG geometric accuracy target: < 0.3 pixels residual (more lenient than Navy's 0.2px because voxel size is 1.0mm vs 0.1mm). | Mean residual < 0.3px confirmed under restricted 180-degree arc with 8 markers. |

### 5.3 NIH Aim 1 Deliverables

| Deliverable ID | Title | Content Description | Pass/Fail Criterion |
|---|---|---|---|
| **NIH-AIM1-01** | Constellation Parameter Grid Study | Run SDSG solver across: marker count N_m = (4, 6, 8) × arc constraint = (restricted 180, full 360). Minimum 6 configurations (3×2). Record mean reprojection residual for each. Store under `aim1/grid_search/`. | Grid results stored. At least 1 config achieves < 0.3px under restricted 180 arc. Best configuration documented. |
| **NIH-AIM1-02** | mART Parameter Sweep | Run mART at iteration counts (25, 50, 100) × relaxation lambda (0.5, 1.0, 2.0) = 9 combinations. Record SSIM, CNR@5mm hemorrhage, and wall-clock time. | 9-row results table stored. Best parameter combination identified. Recommended as default for NIH proposal. |
| **NIH-AIM1-03** | Figure: Constellation Optimization Heatmap | 2D heatmap or grouped bar chart. X-axis: number of markers (4, 6, 8). Y-axis: mean residual (px). Two groups: restricted and full arc. Horizontal dashed line at 0.3px target. Caption: clinical rationale for 8-marker configuration. | Saved as `figures/nih/aim1_fig01_constellation_optimization.png` at 300 DPI. |
| **NIH-AIM1-04** | Figure: mART Convergence Curves | Line plot. X-axis: iteration (0–100). Y-axis: relative update norm (log scale). Three lines for lambda = 0.5, 1.0, 2.0. Recommended lambda annotated. Title: "mART Convergence for Cranial Phantom Reconstruction". | Saved as `figures/nih/aim1_fig02_mart_convergence.png`. All curves decreasing. |
| **NIH-AIM1-05** | Aim 1 Milestone Report Text | Proposal-ready paragraph stating: optimized constellation config (marker count, placement rationale), achieved residual under restricted arc, comparison to full arc, recommended mART parameters. Inserted directly into NIH SBIR Aim 1 methods section. | Residual < 0.3px under restricted arc stated with numerical value. Suitable for direct proposal insertion. |

### 5.4 NIH Aim 2 Preliminary Data Deliverables

| Deliverable ID | Title | Content Description | Pass/Fail Criterion |
|---|---|---|---|
| **NIH-AIM2-01** | Figure: Hemorrhage Detection Slice Comparison | Three-panel axial slice: (A) Ground Truth, (B) FBP restricted arc, (C) mART restricted arc. Soft-tissue window/level. 5mm scale bar. White arrow at hemorrhage in each panel. Caption with CNR values. | 5mm hemorrhage visible in mART panel. Higher SNR than FBP. Saved as `figures/nih/aim2_fig01_hemorrhage_detection.png`. |
| **NIH-AIM2-02** | Figure: CNR vs Lesion Size — Full vs Restricted Arc | Line plot. X-axis: lesion diameter mm (3, 5, 8, 12). Y-axis: CNR. Four lines: FBP full-arc (dashed blue), mART full-arc (solid blue), FBP restricted (dashed red), mART restricted (solid red). Dashed line at CNR=4 labeled "Rose Criterion." | mART restricted arc line >= 4 for 5mm and larger lesions. |
| **NIH-AIM2-03** | Figure: Arc Restriction Artifact Comparison | Four-panel figure: full arc FBP, full arc mART, restricted arc FBP, restricted arc mART. Same slice and window. Demonstrates arc restriction causes streak artifacts in FBP that mART mitigates. | Streak artifacts visible in restricted FBP, reduced in restricted mART. |
| **NIH-AIM2-04** | Table: NIH Image Quality Summary | Columns: Method, Arc, SSIM, PSNR (dB), CNR@3mm, CNR@5mm. Rows: FBP full, mART full, FBP restricted, mART restricted, plus SART variants. | All cells populated. mART restricted arc CNR@5mm >= 4. |
| **NIH-AIM2-05** | Simulated ROC and AUC Estimate | Run 10 lesion-present + 10 no-lesion simulations (different Poisson noise seeds). Compute CNR at hemorrhage. Define positive detection as CNR > threshold T. Sweep T to plot ROC. Compute AUC via trapezoidal integration. **CRITICAL: label in proposal as "computational simulation estimate" — not "clinical diagnostic performance."** | Simulated AUC > 0.75. Saved as `figures/nih/aim2_fig04_simulated_roc.png`. Framing language confirmed honest. |

### 5.5 NIH Proposal Language Requirements

> **Framing (required):** *"Preliminary data presented here constitutes a computational feasibility demonstration using a ray-traced digital cranial phantom with hemorrhage inserts modeled at DeltaHU = 40–50, consistent with the imaging appearance of fresh intracranial blood on non-contrast CT as established in the clinical literature."*

> **Prior validation (required):** *"The SDSG algorithm has been validated by its inventors in non-destructive evaluation of industrial steel specimens (Case and Kenderian, IEEE OJIM, 2023, DOI: 10.1109/OJIM.2023.3268451). This proposal represents the first systematic translation of SDSG to cranial medical imaging."*

> **Clinical feasibility benchmark (required, references Pyakurel 2024):** *"Recent work on compact CBCT for point-of-care brain imaging demonstrates soft-tissue visualization, HU accuracy, contrast resolution, and spatial resolution comparable to gold-standard multidetector CT (Pyakurel et al., Scientific Reports, 2024, DOI: 10.1038/s41598-024-79874-2). Our Phase II prototype targets this same performance bar."*

> **mART vs FBP justification (required, references Sisniega 2015):** *"Iterative reconstruction improves CNR for intracranial hemorrhage detection compared to FBP in cone-beam CT settings, with published improvements from 9.6 (FBP) to 14.2 (iterative) at matched spatial resolution (Sisniega et al., Physics in Medicine and Biology, 2015, PMID 26300578). This motivates our use of mART as the primary reconstruction algorithm."*

> **Positioning against deep learning (required):** *"Unlike markerless deep learning methods (e.g., PICCS, DD-Net) that must simultaneously estimate geometry and reconstruct the image, SDSG resolves shot geometry explicitly from the fiducial constellation before any reconstruction is performed. This decoupling provides geometric certainty under arbitrary freehand ICU scan trajectories, where geometry estimation errors in markerless methods would propagate as reconstruction artifacts indistinguishable from pathology."*

> **Phase II bridge (required):** *"Phase II will procure a CIRS Model 603A anthropomorphic head phantom and conduct 50 physical scan acquisitions (25 lesion-present, 25 lesion-absent) under the locked ICU-constrained access protocol defined in Aim 2, with a five-reader MRMC-ROC blinded evaluation targeting AUC >= 0.85 for 5mm hemorrhage detection."*

---

## 6. Five-Day Execution Plan — April 21–25, 2026

> **Priority:** Navy SBIR (Phantom A) is primary. NIH SBIR (Phantom B) is secondary. Never sacrifice Navy quality for NIH completeness.

### Day 1 — Monday April 21: Environment + Navy Phantom (8 hours)

| Hours | Task | Requirements | Gate Check |
|---|---|---|---|
| 1–2 | Install Miniforge. Create `sdsg_sim` conda env (Python 3.11). Install ASTRA 2.4.1 via conda. Run `astra.test()`. Install pip dependencies. | ENV-001 to ENV-006 | `astra.test()` returns zero failures. All imports succeed. |
| 2–3 | Initialize git repo. Create all folder structure. Commit. Write `environment.yml`. | ENV-005, ENV-006 | `git log` shows initial commit. All folders present. |
| 3–6 | Build Navy steel phantom: 250×250×250 grid, mu_steel=0.1149, three crack planes with partial volume weighting, 6–8 lead spheres at non-coplanar positions, two gas porosity voids. | NV-PHN-001 to NV-PHN-009 | Phantom HDF5 present. Cross-section plots show all three cracks and markers. |
| 6–7 | Visual inspection: matplotlib 3-panel view (axial, coronal, sagittal). Verify crack widths. Verify marker non-coplanarity. | NV-PHN-003 to NV-PHN-007 | All three cracks visible and correctly sized. Markers non-symmetric. |
| 7–8 | Commit phantom builder. Tag v0.1. Update README with phantom parameters. | ENV-005 | Code committed and tagged. |

### Day 2 — Tuesday April 22: Forward Projector + Centroiding (8 hours)

| Hours | Task | Requirements | Gate Check |
|---|---|---|---|
| 1–2 | Build ASTRA `cone_vec` geometry from ground-truth shot positions. Implement per-shot perturbation. Verify sinogram shape `(512, N_shots, 512)` on single test shot. | FWD-001 to FWD-005 | Sinogram shape correct. No ASTRA exceptions. |
| 2–4 | Implement Beer-Lambert conversion, Poisson noise, geometric unsharpness blur. Run full 100-shot forward projection. Save to HDF5. | FWD-005 to FWD-008 | HDF5 sinogram file ~500MB. Ground-truth geometry stored. |
| 4–6 | Build centroiding module: inversion, blob_log, identity matching, center-of-mass refinement. Test on first 10 shots. Debug if identity swaps occur. | CENT-001 to CENT-005 | 10/10 test shots: correct marker count, zero identity swaps. |
| 6–7 | Run centroiding on all 100 shots. Measure noise. Plot histogram of centroiding error. | CENT-005, CENT-006 | Centroiding noise < 0.15px RMS. Histogram saved. |
| 7–8 | Commit forward projector and centroiding modules. Tag v0.2. | ENV-005 | Code committed. |

### Day 3 — Wednesday April 23: SDSG Solver (8 hours — most critical day)

| Hours | Task | Requirements | Gate Check |
|---|---|---|---|
| 1–2 | Implement U7 coordinate translation. Implement pinhole projection model. Implement RMS cost function. **Unit test:** cost function returns < 0.01px when fed ground-truth geometry. | SDSG-001 to SDSG-004 | Unit test passes. Cost function working. |
| 2–4 | Implement LM optimizer call for single U7 problem. Test with first shot, first anchor. Verify convergence in < 100 function evaluations. | SDSG-005, SDSG-006 | Single U7 problem converges. Final residual < 0.2px. |
| 4–5 | Implement anchor iteration loop: cycle through all N_markers anchors per shot. | SDSG-001 | N_markers U7 solutions produced for shot 1. |
| 5–6 | Implement fitness-weighted merge with quaternion rotation averaging. Test on first shot. | SDSG-007 | Merged 9-DOF computed. Quaternion averaging confirmed. |
| 6–7.5 | Run full SDSG solver on all 100 shots. Compute and store residual statistics. | SDSG-008, SDSG-010 | Full run completes without crash. Results stored in HDF5. |
| 7.5–8 | **GATE CHECK:** Verify mean residual < 0.2px under nominal conditions. **If gate fails: debug before advancing to Day 4.** | SDSG-008 | **GATE:** Mean residual < 0.2px confirmed. Tag v0.3. |

### Day 4 — Thursday April 24: Reconstruction + All Navy Deliverables (8 hours)

| Hours | Task | Requirements | Gate Check |
|---|---|---|---|
| 1–1.5 | FBP reconstruction using ASTRA `FDK_CUDA`. Visualize. Note expected artifacts. | RECON-001 | FBP volume produced. Crack faintly visible or blurred. |
| 1.5–3.5 | mART reconstruction: 50 iterations, multiplicative update, col_sum precomputed, epsilon clamping. Run on Navy phantom. | RECON-002 to RECON-006 | mART volume produced. 0.8mm crack visible. No NaN. |
| 3.5–4.5 | Fiducial inpainting. Compute all metrics: SSIM, PSNR, CNR@0.4mm, CNR@0.8mm, CNR@1.6mm for both FBP and mART. | RECON-006, METRIC-001 to METRIC-005 | All metrics computed and stored in HDF5. |
| 4.5–5.5 | Run N=20, 50, 200 additional shot variants. Run stress test (sigma_s=10mm). Store all results. | NV-SHOT-001 to NV-SHOT-004 | Four shot-count datasets complete. Stress test stored. |
| 5.5–8 | Generate all 4 Navy figures (NV-FIG-01 to NV-FIG-04) and 2 tables (NV-TAB-01, NV-TAB-02). Save all at 300 DPI. Review each figure. | NV-FIG-01 to NV-TAB-02 | All 6 Navy deliverables saved. mART CNR@0.8mm >= 4 confirmed. |

### Day 5 — Friday April 25: NIH Phantom + Figures + Documentation (8 hours)

| Hours | Task | Requirements | Gate Check |
|---|---|---|---|
| 1–2 | Build NIH cranial phantom: ellipsoidal skull shell, brain interior, hemorrhage inserts (3/5/8/12mm), edema region, 8 BaSO4 markers. Save lesion and no-lesion variants. | NIH-PHN-001 to NIH-PHN-010 | All 5 phantom HDF5 files present. Cross-section shows skull, brain, and hemorrhage. |
| 2–3 | Configure NIH shot geometry: restricted 180-degree arc, 80 shots, sigma_s=3mm, sigma_theta=1.5deg. Run forward projection and centroiding on NIH phantom. | NIH-SHOT-001 to NIH-SHOT-006 | Centroiding succeeds. Markers correctly detected. |
| 3–4 | Run SDSG solver on NIH phantom (restricted arc). Run constellation grid (6-config minimum). Record residuals. | NIH-AIM1-01, NIH-AIM1-02, SDSG-008 | **GATE:** Mean residual < 0.3px under restricted arc. Grid results stored. |
| 4–5.5 | Run mART on NIH phantom. Run 10 lesion + 10 no-lesion simulations for ROC. Generate NIH figures NIH-AIM1-03, NIH-AIM1-04, NIH-AIM2-01 through NIH-AIM2-04. | NIH-AIM1-03 to NIH-AIM2-04 | All NIH figures saved. Simulated AUC computed. |
| 5.5–6.5 | Compute simulated ROC curve and AUC. Save figure NIH-AIM2-05. | NIH-AIM2-05 | Simulated AUC > 0.75. Figure saved. |
| 6.5–7.5 | Write README.md: environment setup, reproduction instructions per figure, parameter descriptions, limitations statement. | ENV-005 | README complete. Independent reproduction possible from README alone. |
| 7.5–8 | Final checklist review. Verify all deliverable IDs present in `figures/` folder. Commit and tag v1.0. | All deliverable IDs | Git tag v1.0 created. Full checklist confirmed. |

---

## 7. Risk Register and Fallback Protocols

| Risk ID | Risk | Probability | Impact | Fallback |
|---|---|---|---|---|
| RISK-001 | ASTRA CUDA install fails (driver conflict, WSL2 issue) | Medium | High | Install TIGRE package (`pip install pytigre`) — same `cone_vec` API, CPU fallback available. If both fail, reduce to 2D fan-beam slice using ASTRA CPU mode. 2D still demonstrates 07/09 math and produces valid figures. |
| RISK-002 | U7 Levenberg-Marquardt fails to converge under nominal perturbation on Day 3 | Medium | High | Reduce headline perturbation to sigma_s=0.5mm, sigma_theta=0.2deg for Figure 1 numbers. Present 2mm/1deg as "stress test." Add `scipy.optimize.basinhopping` wrapper (5 hops) for shots with residual > 2px. |
| RISK-003 | mART CNR@0.8mm fails to reach 4 (Rose criterion) | Low | Medium | Increase N_shots to 200. Reduce crack size headline to 1.6mm for CNR=4 claim and note 0.8mm as stretch target. Add TV regularization to mART. Adjust proposal language accordingly. |
| RISK-004 | Navy phantom too slow (>8 min per run) blocking Day 4 | Low | Medium | Reduce resolution to 100×100×100 at 0.25mm voxel. Same physical size. Note reduction in proposal. Alternatively run N=50 shots instead of N=100 for reconstruction. |
| RISK-005 | NIH phantom centroiding fails (BaSO4 too low contrast at 70keV) | Medium | Medium | Switch NIH markers to lead (mu=1.133 at 70keV, high contrast). Note in proposal: "Simulation uses lead markers; clinical implementation uses BaSO4 scalp patches." |
| RISK-006 | Day 5 runs out of time before NIH figures complete | Medium | Low | Navy figures (Day 4) are complete. NIH preliminary data section uses Navy geometry results plus 4 NIH-specific citations as feasibility argument. Simulation labeled "planned Month 1 computational work" per Aim 1. |
| RISK-007 | Figure quality poor (small fonts, wrong units, missing labels) | Low | Medium | All figures: set matplotlib rcParams at file top — `figsize=(8,6)`, `fontsize=12`, `linewidth=2.0`, `dpi=300`. Review each figure at 100% zoom before committing. |

---

## 8. Required Citations — Path 1 Literature Foundation

The following citations must be obtained as PDFs, read for the specific quantitative findings listed, and integrated into the proposal text. Each is required — not optional — because it directly addresses a specific reviewer concern identified from the proposal documents.

### 8.1 Citations Required for Both SBIRs

| ID | Full Reference | Source | Required Finding | Claim Supported |
|---|---|---|---|---|
| CIT-001 | Case JT, Kenderian S. Self-Determined Shot Geometry for Open-Configuration Portable X-Ray CT. *IEEE Open J Instrum Meas.* 2023. DOI: 10.1109/OJIM.2023.3268451 | IEEE Xplore (open access). ResearchGate pub 370304009. | Confirm: real X-ray data (not simulation). Quote exact language about geometric error being "satisfactorily low for successful CT reconstruction." | Core SDSG feasibility. Real-data validation by patent inventors. |
| CIT-002 | Case JT, Kenderian S. SDSG for Open-Configuration Portable X-ray CT Using Unknown Constellation. *ASNT RS 2023.* DOI: 10.32548/RS.2023.081 | ResearchGate pub 375437741. | Confirm: markers affixed to specimen and digitally removed from reconstructed volume. | Fiducial digital removal capability claimed in both proposals. |
| CIT-003 | Blumensath T et al. Investigations into Geometric Calibration and Systematic Effects of a Micro-CT System. *Sensors.* 2024;24(16):5139. PMC11360169. | PubMed Central (open access). | Extract residual geometric error values. Confirm range 0.18–0.27px. | Contextualizes Navy <0.2px target as current state of art, not speculative. |
| CIT-004 | ASTM E1742/E1742M-23. Standard Practice for Radiographic Examination. ASTM International. | ASTM.org. Purchase or find accessible copy. | Confirm: 2-2T sensitivity level = 2% of material thickness. At 25mm steel = 0.5mm. | Navy compliance claim. SDSG 0.8mm target provides 60% margin above this standard. |
| CIT-005 | US Patent 12,327,378 B2. Self-Determined Shot Geometry for Open-Configuration Portable X-Ray CT. Assignee: The Aerospace Corporation. | Already in Drive. USPTO public search. | Read claims 1–21. Confirm assignee. Confirm core algorithmic claims match simulation implementation. | All patent-specific statements in both proposals. |

### 8.2 Citations Required for NIH SBIR Only

| ID | Full Reference | Source | Required Finding | Claim Supported |
|---|---|---|---|---|
| CIT-006 | Pyakurel U et al. Evaluation of a compact cone beam CT concept with high image fidelity for point-of-care brain imaging. *Sci Rep.* 2024;14:28286. DOI: 10.1038/s41598-024-79874-2 | Nature.com (open access). | Extract: soft tissue CNR, HU accuracy, spatial resolution metrics vs MDCT benchmark. | Phase II performance target. Establishes portable brain CT can match MDCT. |
| CIT-007 | Sisniega A et al. Cone-Beam CT of Traumatic Brain Injury Using Statistical Reconstruction with Post-Artifact-Correction Noise Model. *Phys Med Biol.* 2015. PMID: 26300578. | PubMed. May need institutional access. | Extract CNR values: 9.6 (FBP) vs 14.2 (iterative) for intracranial hemorrhage. | mART vs FBP justification for brain hemorrhage detection. |
| CIT-008 | Carlson AP, Yonas H. Portable head computed tomography scanner: technology and applications — experience with 3421 scans. *J Neuroimaging.* 2012. PMID: 21699615. | PubMed. | Extract: 3421 scans, 95.8% in neuro ICU, diagnostic adequacy rate. | Clinical need and feasibility precedent for bedside portable head CT. |
| CIT-009 | Wasserthal J et al. TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images. *Radiol Artif Intell.* 2023;5(5):e230024. DOI: 10.1148/ryai.230024 | Radiology AI (open access). | Confirm: provides de-identified skull segmentations from real CT images for simulation digital twin. | Aim 1 methodology — source of skull geometry for digital twin simulation. |

---

## 9. Definition of Done — Acceptance Checklist

The preliminary data work package is complete when every item below is checked. This is the final gate before handing data to Will for proposal integration.

### 9.1 Navy SBIR Acceptance Checklist

| # | Acceptance Item | Status |
|---|---|---|
| 1 | Navy steel phantom HDF5 file present with correct shape (250,250,250) and all datasets | `[ ]` |
| 2 | 100-shot sinogram HDF5 file present with ground-truth geometry per shot | `[ ]` |
| 3 | Centroiding noise < 0.15px RMS confirmed and logged | `[ ]` |
| 4 | SDSG solver mean reprojection residual < 0.2px under nominal conditions confirmed | `[ ]` |
| 5 | Stress test (sigma_s=10mm) results stored separately in HDF5 | `[ ]` |
| 6 | FBP reconstruction volume produced, marker artifacts inpainted | `[ ]` |
| 7 | mART reconstruction volume (50 iterations) produced, marker artifacts inpainted | `[ ]` |
| 8 | All metrics computed: SSIM, PSNR, CNR@0.4mm, CNR@0.8mm, CNR@1.6mm for FBP and mART | `[ ]` |
| 9 | NV-FIG-01: Residual histogram saved at 300 DPI with correct labels and target line | `[ ]` |
| 10 | NV-FIG-02: mART vs FBP slice comparison saved at 300 DPI with scale bar and arrows | `[ ]` |
| 11 | NV-FIG-03: Accuracy vs N_shots line plot saved at 300 DPI | `[ ]` |
| 12 | NV-FIG-04: CNR vs crack width saved at 300 DPI with Rose criterion line | `[ ]` |
| 13 | NV-TAB-01: Shot geometry accuracy table populated with numerical values | `[ ]` |
| 14 | NV-TAB-02: Image quality summary table populated with numerical values | `[ ]` |
| 15 | mART CNR@0.8mm >= 4 (Rose criterion met) | `[ ]` |
| 16 | All 5 shared citations (CIT-001 to CIT-005) obtained as PDFs and key findings extracted | `[ ]` |
| 17 | All 5 proposal framing language bullets (Section 4.4) written and ready for insertion | `[ ]` |

### 9.2 NIH SBIR Acceptance Checklist

| # | Acceptance Item | Status |
|---|---|---|
| 1 | NIH cranial phantom HDF5 files: all 5 variants (lesion 3/5/8/12mm + no-lesion) present | `[ ]` |
| 2 | Restricted 180-degree arc shot configuration run and stored | `[ ]` |
| 3 | SDSG solver mean residual < 0.3px under restricted arc confirmed | `[ ]` |
| 4 | Constellation parameter grid (minimum 6 configurations) run and stored | `[ ]` |
| 5 | mART parameter sweep (minimum 9 configurations) run and stored | `[ ]` |
| 6 | FBP, mART (and SART) reconstructions on NIH phantom produced | `[ ]` |
| 7 | NIH-AIM1-03: Constellation heatmap saved at 300 DPI | `[ ]` |
| 8 | NIH-AIM1-04: mART convergence curves saved at 300 DPI | `[ ]` |
| 9 | NIH-AIM2-01: Hemorrhage detection slice comparison saved at 300 DPI | `[ ]` |
| 10 | NIH-AIM2-02: CNR vs lesion size (4 lines) saved at 300 DPI | `[ ]` |
| 11 | NIH-AIM2-03: Arc comparison 4-panel figure saved at 300 DPI | `[ ]` |
| 12 | NIH-AIM2-04: Image quality table populated | `[ ]` |
| 13 | NIH-AIM2-05: Simulated ROC curve saved. AUC > 0.75 confirmed. | `[ ]` |
| 14 | NIH-specific citations CIT-006 to CIT-009 obtained and findings extracted | `[ ]` |
| 15 | All 6 NIH proposal framing language bullets (Section 5.5) written and ready for insertion | `[ ]` |

### 9.3 Repository and Documentation Checklist

| # | Acceptance Item | Status |
|---|---|---|
| 1 | Git repository with all code committed and tagged v1.0 | `[ ]` |
| 2 | README.md: environment setup, reproduction instructions per figure, parameter table, limitations section | `[ ]` |
| 3 | `requirements.txt` or `environment.yml` present and tested on clean environment | `[ ]` |
| 4 | All figures in `figures/navy/` and `figures/nih/` with filenames matching deliverable IDs | `[ ]` |
| 5 | All HDF5 data organized under `data/navy/` and `data/nih/` | `[ ]` |
| 6 | Known limitations documented: simulation vs real data, no physical hardware validation, Poisson noise model simplified | `[ ]` |
| 7 | Total run time per trial confirmed < 8 minutes for Navy run on target hardware | `[ ]` |

---

## 10. Roles and Responsibilities

| Role | Person | Responsibilities |
|---|---|---|
| **Lead Engineer / Owner** | Edgar J. Suarez Colon | All simulation code. All figure generation. Path 2 execution per 5-day plan. Citation PDF collection and key-finding extraction for Path 1. Maintenance of acceptance checklist. Communication of blockers to PI same day. |
| **PI / Technical Review** | William Rosellini | Review all figures before proposal integration. Confirm proposal framing language is consistent with broader narrative. Final sign-off on acceptance checklist. Not responsible for simulation code. |
| **Algorithm IP Owners** | Case & Kenderian (Aerospace Corp) | Available for technical Q&A on algorithm details if convergence issues arise on Day 3. Not responsible for simulation work or proposal writing. |
| **Proposal Integration** | William Rosellini / Yarira Ortiz | Insert finalized figures and tables into proposal Volume 2 (Navy) and Specific Aims (NIH). Write surrounding narrative using framing language from Sections 4.4 and 5.5 of this PRD as the starting draft. |

---

*CONFIDENTIAL | Darrow Industries, Inc. | SDSG SBIR Preliminary Data PRD v1.0 | April 21, 2026*
