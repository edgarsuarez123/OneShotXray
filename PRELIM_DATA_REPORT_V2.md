# Preliminary Data Report v2 — Darrow Industries SDSG SBIR

**Date:** 2026-05-22  
**Patent:** U.S. 12,327,378 B2 (Self-Determined Shot Geometry, Case & Kenderian, The Aerospace Corporation)  
**Supersedes:** Preliminary Data Report v1.0 (2026-04-24)  
**Repo tag:** v1.1  

---

## 1. Executive Summary

All six go/no-go gates pass for both the Navy and NIH tracks.

Key change from v1.0: the NIH Aim 2 ROC sweep was rerun with per-trial geometry perturbation
(methodology fix detailed in Section 5). The v1.0 AUC of 0.9999 was ceiling-saturated due to a
shared geometry seed across all 200 trials. The corrected AUC is **0.9691**, 95% CI
**[0.9445, 0.9937]** — both gates still pass by comfortable margins.

| Track | Gate | Criterion | Result | Status |
|-------|------|-----------|--------|--------|
| Navy  | Solver residual | < 0.2 px | 0.084 px | **PASS** |
| Navy  | mART CNR @ 0.8 mm | ≥ 4.0 | 14.66 | **PASS** |
| NIH   | Solver residual | < 0.3 px | 0.096 px | **PASS** |
| NIH   | mART CNR @ 5 mm | ≥ 4.0 | 6.94 | **PASS** |
| NIH   | AUC (n=200) | > 0.75 | 0.9691 | **PASS** |
| NIH   | AUC 95% CI lower bound | > 0.85 | 0.9445 | **PASS** |

---

## 2. Methods Summary

### 2.1 Shared SDSG Pipeline

All simulations implement the SDSG algorithm from U.S. Patent 12,327,378 B2:

1. **Forward projection** — ASTRA Toolbox 2.4.1, `cone_vec` geometry type, Beer-Lambert
   attenuation, Poisson noise, Gaussian geometric unsharpness.
2. **Centroiding** — 7-parameter Gaussian fit on raw sinogram patches seeded from GT
   expected positions; co-projecting marker pairs (separation < 14 px) excluded.
3. **SDSG solver** — For each shot: N_markers U7 sub-problems (anchor-7 + 2 unknowns each),
   LM optimizer (`scipy.optimize.least_squares`, method=`lm`, xtol=ftol=1e-10, max_nfev=200,
   x_scale=`jac`); fitness-weighted merge with quaternion rotation averaging (Markley
   eigenvector method); failed U7s (cost > 2.0 px) excluded from merge.
4. **Reconstruction** — mART (multiplicative ART, 25 iter, λ=0.5, ε=1e-6 floor); FBP via
   ASTRA FDK_CUDA (baseline, labeled non-circular).
5. **GPU** — NVIDIA RTX 3080, CUDA 11+.

### 2.2 Navy Track Parameters

| Parameter | Value |
|-----------|-------|
| Energy | 200 keV |
| μ_steel | 0.1149 mm⁻¹ (NIST XCOM iron at 200 keV) |
| μ_lead (markers) | 1.133 mm⁻¹ |
| Voxel size | 0.1 mm isotropic |
| Grid | 250 × 250 × 250 |
| N_shots | 100 (also 20, 50, 200 for sweep) |
| Shot geometry | Fibonacci hemisphere, full 360° azimuth |
| σ_s | 2 mm (position noise std) |
| σ_θ | 1° (orientation noise std) |
| Detector | 512 × 512, 0.2 mm pitch |
| SOD / ODD | 500 mm / 200 mm |
| N_markers | 8 non-coplanar lead spheres |
| Primary crack target | 0.8 mm (ASTM E1742 Rose criterion CNR ≥ 4) |

### 2.3 NIH Track Parameters

| Parameter | Value |
|-----------|-------|
| Energy | 70 keV |
| μ_bone | 0.048 mm⁻¹ |
| μ_brain | 0.021 mm⁻¹ (~40 HU) |
| μ_blood (lesion) | 0.023 mm⁻¹ (ΔHU ≈ 40–50) |
| μ_BaSO4 (markers) | 0.31 mm⁻¹ |
| Voxel size | 1.0 mm isotropic |
| Grid | 200 × 160 × 140 (90 × 70 × 65 mm skull) |
| N_shots | 80 |
| Shot arc | Restricted 180° Fibonacci (ICU access constraint) |
| σ_s | 3 mm |
| σ_θ | 1.5° |
| Detector | 512 × 512, 0.2 mm pitch |
| SOD / ODD | 500 mm / 200 mm |
| N_markers | 8 non-coplanar BaSO4 spheres |
| Primary lesion | 5 mm sphere hemorrhage |

---

## 3. Navy Track Results

Data file: `data/navy/phantom.h5`  
Sinogram: `data/navy/sinogram_100.h5` (shape 512 × 100 × 512, float32)

### 3.1 Geometry Recovery

| Metric | N=20 | N=50 | N=100 | N=200 | Stress N=100 |
|--------|------|------|-------|-------|--------------|
| Mean residual (px) | 0.0862 | 0.0782 | 0.0836 | 0.0776 | 0.0744 |
| Shots solved | 19/20 | 46/50 | 94/100 | 193/200 | 93/100 |

**Gate:** mean residual < 0.2 px — PASS at all shot counts including 10 mm / 5° stress.

Reference figure: `figures/navy/fig01_residual_histogram.png`  
Reference table: `figures/navy/tab01_geometry_accuracy.csv`

### 3.2 Reconstruction Quality

| Method | SSIM | PSNR (dB) | CNR @ 0.8 mm |
|--------|------|-----------|--------------|
| FBP (FDK_CUDA) | 0.00 | −45.25 | 0.35 |
| mART (25 iter, λ=0.5) | 0.23 | 8.65 | **14.66** |

**Gate:** mART CNR @ 0.8 mm = 14.66 ≥ 4.0 — PASS (Rose criterion, ASTM E1742).

mART achieves CNR > 4 at all three crack widths tested (0.4 mm, 0.8 mm, 1.6 mm).  
FBP baseline is non-circular FDK; low metrics are expected and correctly labeled.

Reference figure: `figures/navy/fig02_recon_comparison.png`  
Reference figure: `figures/navy/fig04_cnr_vs_crack.png`  
Reference table: `figures/navy/tab02_image_quality.csv`

---

## 4. NIH Track Results

### 4.1 Aim 1 — Reconstruction Quality

Data files: `data/nih/sinogram_80_restricted.h5`, `data/nih/sinogram_80_full360.h5`

#### 4.1.1 Solver Geometry Recovery

| Arc | Mean residual (px) | Shots solved |
|-----|--------------------|--------------|
| Restricted 180° | 0.0958 | 80/80 |
| Full 360° | 0.0684 | 80/80 |

**Gate:** mean residual < 0.3 px (both arcs) — PASS.

#### 4.1.2 Marker Constellation Sweep

| N_markers | Restricted residual | Full360 residual |
|-----------|---------------------|-----------------|
| 4 | NaN (underdetermined: 8 eqs, 9 DOF) | NaN |
| 6 | 0.053 px | — |
| 8 | 0.096 px | 0.068 px |

8 non-coplanar markers selected as the operational configuration.

Reference figure: `figures/nih/aim1_fig01_constellation_optimization.png`

#### 4.1.3 mART Hyperparameter Sweep

Optimal: **n_iter = 25, λ = 0.5** (CNR = 6.94, SSIM = 0.924, restricted arc).

Reference figure: `figures/nih/aim1_fig02_mart_convergence.png`

#### 4.1.4 Reconstruction Method Comparison (5 mm lesion phantom)

| Method | Arc | SSIM | PSNR (dB) | CNR @ 5 mm |
|--------|-----|------|-----------|------------|
| FBP (FDK_CUDA) | Restricted | 0.014 | −7.80 | 0.18 |
| mART (25, 0.5) | Restricted | **0.924** | **33.35** | **6.94** |
| SART (SIRT3D_CUDA) | Restricted | 0.826 | 34.31 | 3.06 |
| mART (25, 0.5) | Full 360° | 0.936 | 36.07 | 6.53 |

**Gate:** restricted mART CNR @ 5 mm = 6.94 ≥ 4.0 — PASS.

Full 360° is included as the upper-bound reference; restricted arc is the clinically
constrained case (ICU bedside access, 180° aperture).

Reference figure: `figures/nih/aim2_fig01_hemorrhage_detection.png`  
Reference figure: `figures/nih/aim2_fig03_arc_comparison.png`  
Reference table: `figures/nih/aim2_fig04_image_quality.csv`

### 4.2 Aim 2 — Detection Performance (ROC Analysis, v2)

Data file: `data/nih/roc_results.h5` (attrs: `geometry_reseeded=True`)

#### 4.2.1 Trial Design

- 200 independent trials: 100 lesion (5 mm hemorrhage), 100 no-lesion
- Each trial: unique perturbed geometry (geometry_seed = 20000+k), unique Poisson noise
  (seeds 42–141 lesion, 142–241 no-lesion), unique centroid noise (seed = poisson_seed + 10000)
- Decision score: CNR at hemorrhage ROI after SDSG solve + mART reconstruction
- AUC: non-parametric Mann-Whitney estimator
- 95% CI: Hanley-McNeil (1982) analytical formula

#### 4.2.2 Results

| Class | N valid | CNR mean | CNR std |
|-------|---------|----------|---------|
| Lesion | 100/100 | 6.598 | 0.373 |
| No-lesion | 100/100 | 6.055 | 0.174 |

| Statistic | Value |
|-----------|-------|
| AUC | **0.9691** |
| SE | 0.0126 |
| 95% CI | **[0.9445, 0.9937]** |

**Gate 1** (AUC > 0.75): PASS  
**Gate 2** (CI lower bound > 0.85): PASS

Reference figure: `figures/nih/aim2_fig05_simulated_roc.png`

#### 4.2.3 Lesion Size Sensitivity

| Lesion diameter | CNR @ lesion ROI |
|----------------|-----------------|
| 3 mm | 2.84 |
| 5 mm (primary) | 6.94 |
| 8 mm | 11.23 |
| 12 mm | 18.57 |

Reference figure: `figures/nih/aim2_fig02_cnr_vs_lesion.png`

---

## 5. Methodology Note — NIH ROC Geometry Reseed Fix (v1.0 → v2)

### 5.1 Root Cause

The v1.0 ROC script (`analysis/run_nih_roc.py`) called `perturb_geometry(seed=42)` once
before the trial loop, then reused the resulting `gt_9dof` for all 200 trials. Only Poisson
noise and centroid noise varied per trial. Because all 200 reconstructions started from
the same clean sinogram (one fixed geometry), the CNR distributions were artificially narrow:

| Class | v1.0 std | v2 std | Change |
|-------|----------|--------|--------|
| Lesion | 0.139 | 0.373 | +168% |
| No-lesion | 0.125 | 0.174 | +39% |

With a near-degenerate distribution, the Mann-Whitney AUC saturated at 0.9999 and the
Hanley-McNeil CI overflowed above 1.0 ([0.9985, 1.0013]).

### 5.2 Fix

`run_trial()` now draws a fresh geometry perturbation inside each trial:

```python
gt_9dof    = perturb_geometry(nominal_src, SOD, ODD, SIGMA_S, SIGMA_THETA,
                              seed=geometry_seed)
vectors_gt = geometry_to_cone_vec(gt_9dof, DET_SPACING, DET_ROWS, DET_COLS)
clean_sino = forward_project(vol, vectors_gt, VOXEL_SIZE, DET_ROWS, DET_COLS)
```

Seed assignments (disjoint ranges):
- `geometry_seed`: 20000–20099 (lesion), 20100–20199 (no-lesion)
- Poisson seed: 42–141 (lesion), 142–241 (no-lesion)
- Centroid seed: poisson_seed + 10000

This correctly models trial-to-trial freehand scan variability (different session or
operator), consistent with the σ_s = 3 mm, σ_θ = 1.5° specifications in the SBIR proposal
narrative. The inter-trial geometry difference at a single shot has std = √2 · σ ≈ 4.2 mm
per axis and ≈ 2.1° per axis.

### 5.3 Compute Cost

Each additional `forward_project` call adds ~0.3 s on RTX-3080. Total overhead for 200 trials:
~60 s (< 2% of the 93.3-min sweep). Wall time v2: 93.3 min vs v1.0: 90.6 min.

---

## 6. Reproducibility

```powershell
# Full NIH pipeline from scratch:
.\run.ps1 phantom\nih_phantom.py
.\run.ps1 forward\run_nih.py
.\run.ps1 solver\run_nih_solver.py --arc both
.\run.ps1 recon\run_nih_recon.py --step all
.\run.ps1 analysis\run_nih_roc.py          # ~93 min GPU
.\run.ps1 analysis\figures_nih.py

# Resume a crashed ROC sweep:
.\run.ps1 analysis\run_nih_roc.py --resume

# Full Navy pipeline:
.\run.ps1 phantom\navy_phantom.py
.\run.ps1 forward\run_navy.py
.\run.ps1 forward\run_navy_variants.py
.\run.ps1 solver\run_solver.py
.\run.ps1 recon\run_navy_recon.py
.\run.ps1 analysis\figures_navy.py
```

**Environment:** conda env `sdsg_sim` (Python 3.11, ASTRA 2.4.1 CUDA, NumPy ≥ 1.26,
SciPy ≥ 1.13, scikit-image ≥ 0.22, matplotlib ≥ 3.8, h5py ≥ 3.10).  
**GPU:** NVIDIA RTX 3080, CUDA 11+, 10 GB VRAM.  
**Tag:** v1.1 — all results reproducible from this tag.

---

## 7. Figure Index

| Filename | Content | Section |
|----------|---------|---------|
| `figures/navy/fig01_residual_histogram.png` | SDSG solver residual distribution (100 shots) | 3.1 |
| `figures/navy/fig02_recon_comparison.png` | FBP vs mART reconstruction slices at crack | 3.2 |
| `figures/navy/fig03_accuracy_vs_shots.png` | Mean residual vs N_shots (20/50/100/200) | 3.1 |
| `figures/navy/fig04_cnr_vs_crack.png` | CNR vs crack width (0.4/0.8/1.6 mm) | 3.2 |
| `figures/nih/aim1_fig01_constellation_optimization.png` | Residual vs marker count × arc | 4.1.2 |
| `figures/nih/aim1_fig02_mart_convergence.png` | mART convergence curves (λ sweep) | 4.1.3 |
| `figures/nih/aim2_fig01_hemorrhage_detection.png` | GT / FBP / mART slices at hemorrhage | 4.1.4 |
| `figures/nih/aim2_fig02_cnr_vs_lesion.png` | CNR vs lesion diameter (3/5/8/12 mm) | 4.2.3 |
| `figures/nih/aim2_fig03_arc_comparison.png` | Restricted vs full360 × method comparison | 4.1.4 |
| `figures/nih/aim2_fig05_simulated_roc.png` | ROC curve (n=200) with CI band + density inset | 4.2.2 |
