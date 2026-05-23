# OneShotXray — SDSG SBIR Preliminary Data

Computational preliminary data for two concurrent SBIR Phase I proposals by Darrow Industries, Inc.,
implementing U.S. Patent 12,327,378 B2 (Self-Determined Shot Geometry, Case & Kenderian,
The Aerospace Corporation).

- **Navy DON26BZ01-NV012** — SDSG for 3D characterization of ship hull fatigue/corrosion (steel, 200 keV)
- **NIH NIBIB** — Bedside brain hemorrhage monitoring via open-configuration portable CT (cranial, 70 keV)

---

## Environment Setup

```powershell
# Install Miniforge3 then:
conda env create -f environment.yml
# Verify GPU:
.\run.ps1 -c "import astra; astra.test()"
```

**Required:** NVIDIA GPU with CUDA 11+, minimum 8 GB VRAM.

All scripts run through `.\run.ps1 <script.py>` which sets the conda env PATH so ASTRA CUDA DLLs load.

---

## Reproducing Navy Results

```powershell
# 1. Build phantom
.\run.ps1 phantom\navy_phantom.py

# 2. Forward project (100 shots + variants)
.\run.ps1 forward\run_navy.py
.\run.ps1 forward\run_navy_variants.py

# 3. Run SDSG solver
.\run.ps1 solver\run_solver.py

# 4. Reconstruct (FBP + mART)
.\run.ps1 recon\run_navy_recon.py

# 5. Generate all Navy figures
.\run.ps1 analysis\figures_navy.py
```

Output figures: `figures/navy/fig01_residual_histogram.png` through `fig04_cnr_vs_crack.png`

---

## Reproducing NIH Results

```powershell
# 1. Build 5 phantom variants (5mm, 3mm, 8mm, 12mm lesion + no-lesion)
.\run.ps1 phantom\nih_phantom.py

# 2. Forward project both arcs (restricted 180 deg + full 360 deg)
.\run.ps1 forward\run_nih.py

# 3. Run SDSG solver on both arcs
.\run.ps1 solver\run_nih_solver.py --arc both

# 4. Reconstruct + Aim 1 sweeps (~30-60 min GPU)
.\run.ps1 recon\run_nih_recon.py --step all

# 5. Run n=200 ROC sweep (~90 min GPU)
.\run.ps1 analysis\run_nih_roc.py

# 6. Generate all NIH figures
.\run.ps1 analysis\figures_nih.py
```

Output figures: `figures/nih/aim1_fig01_*.png` through `aim2_fig05_*.png`

To resume a crashed ROC sweep: `.\run.ps1 analysis\run_nih_roc.py --resume`

---

## Key Results

### Navy Track

| Method | SSIM | PSNR (dB) | CNR@0.8mm |
|--------|------|-----------|-----------|
| FBP    | 0.00 | -45.25    | 0.35      |
| mART   | 0.23 | 8.65      | **14.66** |

**Gate PASS:** mART CNR@0.8mm = 14.66 >= 4.0 (Rose criterion)

Solver residual: 0.0836 px mean (94/100 shots solved) — gate < 0.2 px

### NIH Track

| Method | Arc        | SSIM  | PSNR (dB) | CNR@5mm |
|--------|------------|-------|-----------|---------|
| FBP    | restricted | 0.014 | -7.80     | 0.18    |
| mART   | restricted | 0.924 | 33.35     | **6.94** |
| SART   | restricted | 0.002 | -17.60    | 1.10    |
| mART   | full360    | 0.936 | 36.07     | 6.53    |

**Gate PASS:** restricted mART CNR@5mm = 6.94 >= 4.0 (Rose criterion)

Solver residual: 0.096 px (restricted), 0.068 px (full360) — gate < 0.3 px

Aim 1 constellation: 4 markers underdetermined, 6 markers 0.053 px, 8 markers 0.096 px

Aim 1 mART sweep: optimal at n_iter=25, lam=0.5 (CNR=6.94, SSIM=0.924)

ROC AUC (n=200, per-trial geometry): 0.9691, 95% CI [0.9445, 0.9937]

---

## Project Structure

```
phantom/        -- phantom builders (Navy steel + NIH cranial)
forward/        -- ASTRA cone_vec forward projector + noise models
solver/         -- SDSG solver (U7 problems, LM optimizer, quaternion merge)
recon/          -- FBP (FDK_CUDA), mART, SART
analysis/       -- metrics (SSIM, PSNR, CNR, ROC), figure generators
data/navy/      -- Navy HDF5 sinograms and results
data/nih/       -- NIH HDF5 phantom variants, sinograms, and results
figures/navy/   -- Navy proposal figures (300 DPI)
figures/nih/    -- NIH proposal figures (300 DPI)
```

---

## Gates Summary

| Gate | Criterion | Result |
|------|-----------|--------|
| Navy solver residual | < 0.2 px | 0.084 px PASS |
| Navy mART CNR@0.8mm | >= 4.0 | 14.66 PASS |
| NIH solver residual | < 0.3 px | 0.096 px PASS |
| NIH mART CNR@5mm | >= 4.0 | 6.94 PASS |
| NIH AUC (n=200) | > 0.75 | 0.9691 PASS |
| NIH AUC CI lower bound | > 0.85 | 0.9445 PASS |
