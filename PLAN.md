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
- [ ] 1. U7 coordinate translation + pinhole projection model + cost function
- [ ] 2. LM optimizer for single U7 problem — unit test < 0.01px at ground truth
- [ ] 3. Anchor iteration loop (N_markers U7 problems per shot)
- [ ] 4. Fitness-weighted merge with quaternion rotation averaging
- [ ] 5. Full 100-shot solver run — store residuals in HDF5
- [ ] **GATE:** Mean residual < 0.2px — must pass before Day 4
- [ ] 6. Commit + tag v0.3

## Day 4 — Reconstruction + All Navy Deliverables
- [ ] 1. FBP via ASTRA FDK_CUDA
- [ ] 2. mART (50 iter, multiplicative, col_sum precomputed, epsilon=1e-6)
- [ ] 3. Fiducial inpainting + compute metrics (SSIM, PSNR, CNR@0.4/0.8/1.6mm)
- [ ] 4. N=20,50,200 shot variants + stress test (sigma_s=10mm)
- [ ] 5. Generate NV-FIG-01 through NV-FIG-04 + NV-TAB-01, NV-TAB-02 at 300 DPI
- [ ] **GATE:** mART CNR@0.8mm >= 4 (Rose criterion)

## Day 5 — NIH Phantom + Figures + Documentation
- [ ] 1. NIH cranial phantom (5 variants) → data/nih/
- [ ] 2. NIH forward projection + centroiding (restricted 180-degree arc, 80 shots)
- [ ] 3. SDSG solver on NIH — GATE: residual < 0.3px
- [ ] 4. Aim 1 constellation grid + mART sweep
- [ ] 5. mART on NIH + 10 lesion + 10 no-lesion ROC simulations
- [ ] 6. Generate all NIH figures (NIH-AIM1-03/04, NIH-AIM2-01 through NIH-AIM2-05)
- [ ] 7. README.md (environment setup, reproduction instructions per figure)
- [ ] 8. Final checklist, commit + tag v1.0

---

## Resume From Here
**Last completed:** Day 2 ALL steps done. sinogram_100.h5 saved with corrected centroids.
**Status:** ALL Day 2 checks PASS — ready for Day 3 (SDSG Solver).

**CENT-005 fix summary:** 3 root causes fixed in centroiding.py:
1. GT pixel center: N/2 → (N-1)/2 per ASTRA convention (eliminated 0.5px systematic)
2. 7-param Gaussian fit on raw sinogram (eliminated cross-marker background contamination)
3. min_separation_px: 8 → 14px (eliminated PSF-overlap errors from near-neighbor pairs)

**Next step (resume here):** Day 3 Step 1 — SDSG Solver
  - U7 coordinate translation + pinhole projection model + cost function
  - Input: sinogram_100.h5 centroids/positions (764 valid detections, 0.1274px noise)
  - Target: solver residual < 0.2px per shot
