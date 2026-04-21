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
- [ ] 1. Build ASTRA cone_vec geometry from ground-truth shot positions
- [ ] 2. Per-shot perturbation (sigma_s, sigma_theta), Beer-Lambert, Poisson noise, unsharpness blur
- [ ] 3. Run 100-shot forward projection → HDF5 sinogram
- [ ] 4. Centroiding module: blob_log + center_of_mass refinement
- [ ] 5. Run all 100 shots, measure centroiding noise < 0.15px RMS
- [ ] 6. Commit + tag v0.2

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
**Last completed:** Day 1 complete — phantom built and inspected, tagged v0.1
**Next step:** Day 2, Step 1 — Build ASTRA cone_vec forward projector
