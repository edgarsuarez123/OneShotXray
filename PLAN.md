# SDSG SBIR — Execution Plan
**Deadline:** April 29, 2026 | **Sprint:** April 21–25, 2026
**Priority:** Navy SBIR (Phantom A) first. Never sacrifice Navy for NIH completeness.

---

## Day 1 — Environment + Navy Phantom

- [x] 1. Git init + folder structure — done 2026-04-21
- [x] 2. conda env sdsg_sim + ASTRA 2.4.1 (CUDA) + all deps — done 2026-04-21
      NOTE: pip ASTRA fails (source build, no CUDA toolkit). Installed Miniforge3 to C:\Users\Edgar\miniforge3.
      Run scripts via: .\run.ps1 <script.py> (sets Library\bin PATH for CUDA DLLs)
- [x] 3. ASTRA.test() PASS — RTX 3080 detected, CUDA 2D+3D OK — done 2026-04-21
- [x] 4. environment.yml committed — done 2026-04-21
- [x] 5. Navy phantom built (250^3, 3 cracks PV-weighted, 8 markers non-coplanar, 2 porosity voids) — done 2026-04-21
- [x] 6. Inspection figure saved → figures/navy/phantom_inspection.png — done 2026-04-21
- [ ] 7. Commit + tag v0.1

## Day 2 — Forward Projector + Centroiding
- [x] 1. Write forward/geometry.py — done 2026-04-22
- [x] 2. Write forward/projector.py — done 2026-04-22
- [x] 3. Write forward/centroiding.py — done 2026-04-22 (iterations ongoing for noise target)
- [x] 4. Write forward/run_navy.py + save data/navy/sinogram_100.h5 — done 2026-04-22
- [x] 5. Verify centroid noise <0.15px RMS — done 2026-04-22 (0.1274px, 95.5% detect rate) PASS
- [x] 6. Commit + tag v0.2 — done 2026-04-22

### Day 2 Verified (PASS)
- sinogram shape (512, 100, 512) ✓ FWD-008
- Central pixel shot 0 = 2.873 (expected ~2.87 for 25mm steel at 200keV) ✓
- Detection rate 98.4% (492/800 of 100 shots x 8 markers) ✓ ≥95% CENT-002
- GT 9-DOF shape (100, 9) ✓
- Zero identity swap errors ✓ CENT-003

### Day 2 BLOCKER — Centroiding Noise CENT-005
**Target:** < 0.15px RMS per shot
**Current:** ~0.65-0.75px mean, max 2.65px

Root causes diagnosed via diag_centroid.py / diag2_permarker.py / diag3_formula.py:

1. **GT pixel offset bug** (fixed in GT formula): was `det_cols/2 = 256`, should be
   `(det_cols-1)/2 = 255.5` per ASTRA convention. Eliminated 0.5px systematic bias.

2. **Cross-marker background contamination** (PRIMARY cause of ~0.6px residual):
   Background subtraction with sigma=30px at a marker location is contaminated by
   neighboring markers projecting within ~45px (average spacing in this geometry).
   Gaussian kernel at 45px: exp(-45^2/(2*30^2)) = 0.32 — creates 0.3-0.5px gradient bias.

3. **Horizontal shot disambiguation failure** (8 shots, cos_theta < 0.1):
   Markers co-project within 2-5px — Gaussian fit picks wrong neighbor's blob.
   Contributes 2-6px errors for these shots.

**Next approach:** 7-parameter Gaussian fit (Gaussian + linear background gradient) on the
raw sinogram. Linear gradient term absorbs cross-marker contamination, eliminating bias
without needing background subtraction entirely. Also fix GT formula to use (N-1)/2 offset.

### Architectural decisions (session 2026-04-22)
- ASTRA volume axis order: (Z,Y,X) = volume.transpose(2,1,0) from phantom's (X,Y,Z) indexing
- vol_geom extents in mm: create_vol_geom(250,250,250,-12.5,12.5,-12.5,12.5,-12.5,12.5)
- All cone_vec coordinates in mm; ASTRA computes path lengths in mm → sino dimensionless ✓
- u = column direction (horizontal), v = row direction (vertical), both scaled by det_spacing
- Unsharpness sigma = 1 pixel = 0.2mm (PRD stated value; formula gives 0.085mm but PRD wins)
- Poisson via np.random.default_rng(42) for bit-identical reproducibility
- Blob detection: background-subtract sino (sigma=30), normalize, blob_log threshold=0.1
- Marker matching: one-to-one greedy sort by distance, max_dist=15px
- Unsharpness weights: w = 1/(1 + Ug/pixel_size), Ug computed per-shot per-marker from geometry
- GT projection formula: uses ASTRA cone_vec u,v vectors; pixel center at (N-1)/2 (CORRECTED)

## Day 3 — SDSG Solver (most critical)
- [x] 1. U7 coordinate translation + pinhole projection model + cost function — done 2026-04-23
- [x] 2. LM optimizer for single U7 problem — unit test < 0.01px at ground truth — done 2026-04-23
- [x] 3. Anchor iteration loop (N_markers U7 problems per shot) — done 2026-04-23
- [x] 4. Fitness-weighted merge with quaternion rotation averaging — done 2026-04-23
- [x] 5. Full 100-shot solver run — store residuals in HDF5 — done 2026-04-23
- [x] **GATE:** Mean residual 0.0836px < 0.2px — PASS 2026-04-23 ✓
- [x] 6. Fix scipy Euler convention bug (R_to_euler + euler_to_quat) — done 2026-04-23
- [x] 7. All 18 unit tests pass — done 2026-04-23
- [x] 8. Commit + tag v0.3 — done 2026-04-23 (tag local; remote tag push blocked 403)
- [x] 9. PR review + gate run + two bug fixes — done 2026-04-23
       Bug 1: Fixed per-shot initial guess (was fixed [0,0,-SOD] for all shots;
              shots are on Fibonacci hemisphere so source can be 700+mm from initial
              guess — LM never converges). Fix: use nominal_src per shot.
              Effect: 743→61 U7 failures, 205s→22s.
       Bug 2: All-failed shots (4 shots, exactly 4 detections, LM diverges to
              degenerate geometry) returned garbage fallback and 88-147px residuals,
              pulling nanmean from 0.084→5px. Fix: return NaN residuals when all
              detected-anchor U7s fail, exclude from mean.

## Day 4 — Reconstruction + All Navy Deliverables
- [x] 0. All Day 4 code written (recon/, analysis/, forward/run_navy_variants.py) — done 2026-04-24
- [x] 1. NV-FIG-01 residual histogram — done 2026-04-24 ✓
- [x] 2. FBP + mART (50 iter) + inpainting + metrics — done 2026-04-24 (81s)
- [x] **GATE:** mART CNR@0.8mm = 14.66 >= 4 — PASS 2026-04-24 ✓
- [x] 3. N=20,50,200 shot variants + stress test — done 2026-04-24
- [x] 4. NV-FIG-01..04 + NV-TAB-01..02 all generated — done 2026-04-24 ✓

## Day 5 — NIH Rebuild Sprint (recovery from lost branch + scaled ROC)

> NOTE 2026-05-22: Day 5 was implemented April 24 per the data report but never committed.
> Code and HDF5 data are unrecoverable. Rebuilding from scratch. ROC also scaled up from
> n=20 → n=200 (100 lesion + 100 no-lesion) for a tighter AUC CI before submission.

### 5.0 — Infrastructure prerequisites
- [ ] 1. Verify conda env `sdsg_sim` and `astra.test()` PASS on current machine
- [ ] 2. Modify `recon/mart.py`: add `lam: float = 1.0` + optional `grid_nx/grid_ny/grid_nz`
         for non-cubic grids (NIH is 200×160×140). Cubic-grid default unchanged.
- [ ] 3. Write `recon/sart.py` — SART 50 iter, λ=1.0 (NIH algorithm comparison)
- [ ] 4. Add NIH masks to `analysis/metrics.py`: `build_nih_hemorrhage_mask()` (sphere r=4mm
         at (58,20,0)mm) and `build_nih_bg_mask()` (sphere r=8mm at (-58,20,0)mm)

### 5.1 — NIH phantom builder
- [ ] 1. Write `phantom/nih_phantom.py`
         Grid 200×160×140 at 1.0mm/voxel. Skull: ellipsoid semi-axes (90,70,65)mm, 7mm shell,
         μ_bone=0.048. Brain μ=0.021; edema contralateral r=15mm μ=0.019; hemorrhage sphere at
         (58,20,0)mm configurable diameter μ=0.023. 8 BaSO₄ markers (r=1mm, μ=0.310) on skull
         surface, non-coplanar verified. API: `build_nih_phantom(lesion_mm) → (vol, markers)`
- [ ] 2. Build 5 variants → data/nih/phantom_{lesion_3mm,lesion_5mm,lesion_8mm,lesion_12mm,nolesion}.h5
- [ ] 3. Save figures/nih/phantom_inspection.png

### 5.2 — Forward projection + CRB centroid model
- [ ] 1. Write `forward/nih_geometry.py`: `generate_restricted_arc_shots(n=80, sod=500)`
         — Fibonacci sampling with phi ∈ [0,π] (lateral 180° arc, ICU bedside constraint).
         Reuse `perturb_geometry(σ_s=3mm, σ_θ=1.5°)` and `geometry_to_cone_vec` from geometry.py.
- [ ] 2. Write `forward/nih_centroiding.py`: CRB noise σ=0.10px on GT projections from
         `solver.projection.project_points_batch`. Skull gradient at 70keV defeats Gaussian
         fitting — simulated noise is the physically-justified approach (report §3.3).
         Detection mask: in-bounds = within [1, 510]×[1, 510] pixels.
- [ ] 3. Write `forward/run_nih.py`: 512×512 det at 0.4mm pitch, I₀=10,000, 70keV.
         Reuse `forward_project` + `apply_noise_pipeline` from projector.py.
         Output: data/nih/sinogram_80_restricted.h5, data/nih/sinogram_80_full360.h5

### 5.3 — SDSG solver on NIH
- [ ] 1. Write `solver/run_nih_solver.py` (mirror run_solver.py, reuse `solve_shot` unchanged)
- [ ] **GATE:** mean residual < 0.3 px

### 5.4 — Aim 1: constellation grid + mART parameter sweep
- [ ] 1. Constellation grid: markers {4,6,8} × arc {restricted, full360} = 6 configs
         → data/nih/aim1_constellation_grid.h5
- [ ] 2. mART sweep: iterations {25,50,100} × λ {0.5,1.0,2.0} = 9 configs
         → data/nih/aim1_mart_sweep.h5
- [ ] 3. Figures: aim1_fig01_constellation_optimization.png, aim1_fig02_mart_convergence.png

### 5.5 — Aim 2: reconstruction + CNR vs lesion size
- [ ] 1. Primary recon: 5mm lesion × {restricted,full360} × {FBP,mART,SART} = 6 runs
         → data/nih/recon_*.h5
- [ ] **GATE:** restricted mART CNR@5mm ≥ 4 (Rose criterion)
- [ ] 2. CNR vs lesion: {3,5,8,12}mm × {restricted,full360} × mART = 8 runs
         → data/nih/recon_lesion_sweep.h5
- [ ] 3. Figures: aim2_fig01_hemorrhage_detection.png, aim2_fig02_cnr_vs_lesion.png,
         aim2_fig03_arc_comparison.png, aim2_fig04_image_quality.csv

### 5.6 — Aim 2: scaled ROC (n=200)
- [ ] 1. Write `analysis/run_nih_roc.py`
         Optimisation: forward-project clean sinogram once per phantom; re-apply Poisson noise
         per trial (cheap). Solver + mART rerun each trial (expensive ~25s each).
         Incremental HDF5 writes + `--resume` flag so a crash mid-run loses no work.
- [ ] 2. Run 100 lesion trials (seeds 42–141) + 100 no-lesion (seeds 142–241)
         → data/nih/roc_results.h5 with cnr_lesion[100], cnr_nolesion[100]
- [ ] 3. AUC via Mann-Whitney (pure numpy, no sklearn). 95% CI via Hanley-McNeil formula.
- [ ] **GATE 1:** AUC > 0.75
- [ ] **GATE 2:** AUC 95% CI lower bound > 0.85
         (n=200 drops SE from ≈0.060 to ≈0.019 vs original n=20 design)
- [ ] 4. Figure: aim2_fig05_simulated_roc.png — ROC curve with CI band + CNR score density inset

### 5.7 — Documentation + v1.0
- [ ] 1. Write README.md (env setup, per-figure reproduction commands)
- [ ] 2. Update progress.txt with all Day 5 step results
- [ ] 3. Confirm all 4 gates passed: solver < 0.3px | CNR ≥ 4 | AUC > 0.75 | CI_low > 0.85
- [ ] 4. `git commit` + `git tag v1.0` + `git push origin master --tags`

---

## Resume From Here
**Last completed:** Day 4 merged from origin/claude/plan-next-priorities-Cy0Iw → master, pushed
                    to origin/master on 2026-05-22. Navy track fully complete.
**Lost work:** Day 5 (NIH) never committed. Code + HDF5 data unrecoverable. Rebuilding.
**New requirement:** ROC n=20 → n=200. New gate: AUC 95% CI lower bound > 0.85.

**Day 4 results (for reference):**
- FBP: SSIM=0.00, PSNR=-45.25dB, CNR@0.8mm=0.35 (expected for non-circular orbit)
- mART: SSIM=0.23, PSNR=8.65dB, CNR@0.8mm=14.66 (GATE PASS >> 4)
- Variants all < 0.09px; stress (σ_s=10mm) = 0.074px
- Figures: fig01..04 + tab01..02 in figures/navy/ at 300 DPI

**Next step (resume here):** §5.0.1 — run `.\run.ps1 -c "import astra; astra.test()"` to
confirm GPU is available on the current machine before writing any NIH code.

**Estimated remaining effort:** ~16 hours coding + ~2 hours GPU compute for n=200 ROC sweep.
