# CLAUDE.md — Edgar J. Suárez Colón

> Master context file for Claude Code. Read this entire file before doing anything else in any session.

---

## Who I Am

I'm a computer engineering student and entrepreneur. I start the U.S. Air Force Palace Acquire (PAQ) program in May 2026, placed in the Weather Systems branch working on cloud infrastructure, distributed systems, and data pipelines. I operate primarily from my phone via SSH/Termius into this PC. I have limited time so efficiency and autonomous execution are critical.

**Long-term goals:**
- Fractional CTO status
- Consulting practice with equity-based compensation
- Recurring SaaS revenue targeting $15K MRR

---

## How I Work

I connect remotely via phone. Sessions can drop at any time. You must operate with maximum autonomy and resilience.

### Branch-Specific Plan Files — ALWAYS FOLLOW THIS

This is a single-sprint repo (5-day execution, April 21–25). One plan file on `main`.

| Branch | Plan File |
|--------|-----------|
| `main` | `PLAN.md` |

Check your current branch at session start and only read/update that branch's plan file.

### Session Resilience Rules — ALWAYS FOLLOW THESE

- At the start of every task, update the **current branch's plan file** (see table above)
- Break every task into numbered steps in the plan file
- Mark each step complete with a timestamp as you finish it: `- [x] 1. Step name — done 06:12`
- Commit to git after every completed step with a clear descriptive message
- Before any large change, commit the current state first as a checkpoint
- If you hit a blocker, document it in the plan file and move to the next step
- Never stop working because of a minor uncertainty — make a reasonable decision, document it, and continue
- At the end of every session update PLAN.md with a "Resume From Here" section
- Update `progress.txt` after each completed step — not just at the end

### How to Resume a Session

If I say "read PLAN.md and continue" — do exactly that. Read the plan, identify the first incomplete step, and resume without asking me to re-explain anything.

### Running Background Processes (Windows — No tmux)

On this machine, use PowerShell `Start-Process` to detach long-running jobs so they survive SSH disconnect:

```powershell
Start-Process -WindowStyle Hidden "conda" `
  -ArgumentList "run -n sdsg_sim python phantom/build_navy_phantom.py" `
  -WorkingDirectory "C:\Users\Edgar\OneShotXRay" `
  -RedirectStandardOutput "C:\Users\Edgar\OneShotXRay\phantom_build.log" `
  -RedirectStandardError "C:\Users\Edgar\OneShotXRay\phantom_build_err.log"
```

Check progress: `Get-Content "C:\Users\Edgar\OneShotXRay\phantom_build_err.log" -Tail 5`

---

## This Project — SDSG SBIR Preliminary Data

**What it is:** Computational preliminary data generation for two concurrent SBIR Phase I proposals submitted by Darrow Industries, Inc. Both proposals implement U.S. Patent 12,327,378 B2 — the Self-Determined Shot Geometry (SDSG) system by Case and Kenderian of The Aerospace Corporation.

**Two tracks:**
- **Navy DON26BZ01-NV012** — SDSG for 3D characterization of ship hull fatigue and corrosion (steel, 200keV)
- **NIH NIBIB** — Bedside brain hemorrhage monitoring via open-configuration portable CT (cranial, 70keV)

**Deadline:** April 29, 2026 (both proposals). 5-day execution sprint April 21–25.

**Key files:**
- `SDSG_SBIR_Preliminary_Data_PRD.md` — full requirements, acceptance checklists, deliverable specs
- `PLAN.md` — current task plan (always maintain this)
- `progress.txt` — running implementation log
- `data/navy/phantom.h5` — Navy steel phantom (250×250×250, 0.1mm voxels)
- `data/nih/` — NIH cranial phantom variants (5mm lesion primary, no-lesion, 3/8/12mm)
- `figures/navy/` — NV-FIG-01 through NV-FIG-04 deliverables
- `figures/nih/` — NIH-AIM1 and NIH-AIM2 figure deliverables

**Folder structure:**
```
phantom/     — phantom builders (Navy steel + NIH cranial)
forward/     — ASTRA cone_vec forward projector + noise models
solver/      — SDSG 07/09 solver, U7 problems, quaternion merge
recon/       — FBP (FDK_CUDA), mART, SART, inpainting
analysis/    — metrics (SSIM, PSNR, CNR, ROC), figure generators
data/navy/   — HDF5 sinograms, geometry, results (Navy)
data/nih/    — HDF5 sinograms, geometry, results (NIH)
figures/     — all proposal-ready figures at 300 DPI
notebooks/   — exploratory / debugging notebooks
```

### Environment

- **OS:** Windows 10 Pro
- **Shell:** PowerShell (SSH via Termius/Tailscale at `100.87.72.29`)
- **Python:** conda env `sdsg_sim` (Python 3.11) — always use this, NOT system Python
- **Activate:** `conda activate sdsg_sim`
- **Run any script:** `conda run -n sdsg_sim python <module>`
- **Verify GPU:** `conda run -n sdsg_sim python -c "import astra; astra.test()"`
- **Install deps:** `conda env create -f environment.yml` (or `conda install -c astra-toolbox astra-toolbox=2.4.1`)

**Required packages:** ASTRA Toolbox 2.4.1 (CUDA via conda), NumPy >= 1.26, SciPy >= 1.13, scikit-image >= 0.22, matplotlib >= 3.8, h5py >= 3.10, tqdm >= 4.66

**GPU requirement:** NVIDIA with CUDA 11+, minimum 8GB VRAM. Fallback: TIGRE package CPU mode.

### Current State (as of 2026-04-21)

- Day 1 of 5-day sprint. Repo initialized, PRD complete.
- No simulation code yet — building from scratch per PRD.
- Navy SBIR is primary priority. Never sacrifice Navy quality for NIH completeness.
- After Day 4 Navy figures are done, Day 5 handles NIH.

### Architecture — SDSG Simulation Pipeline

```
Phantom Builder
  → 3D volume (HDF5, float32)

Forward Projector  [forward/]
  → ASTRA cone_vec geometry (per-shot 9-DOF, ground truth stored)
  → Beer-Lambert + Poisson noise + geometric unsharpness
  → Sinogram HDF5 shape: (det_rows=512, N_shots, det_cols=512)

Centroiding Module  [forward/]
  → blob_log coarse detection (inverted projections)
  → center_of_mass sub-pixel refinement
  → Target: < 0.15px RMS centroiding noise

SDSG Solver  [solver/]
  → 07/09 anchor split: N_markers U7 problems per shot
  → LM optimizer (scipy.optimize.least_squares, method='lm')
  → Fitness-weighted merge with quaternion rotation averaging
  → Target: < 0.2px residual (Navy), < 0.3px (NIH)

Reconstruction  [recon/]
  → FBP: ASTRA FDK_CUDA (baseline, labeled non-circular)
  → mART: 50 iter multiplicative update, epsilon=1e-6, col_sum precomputed
  → SART: NIH only
  → Fiducial inpainting: 1.5mm radius, 3mm annular shell mean

Metrics + Figures  [analysis/]
  → SSIM, PSNR (data_range explicit), CNR (defect + bg ROI)
  → Position/angular recovery error per shot
  → Simulated ROC + AUC (NIH Aim 2)
```

### Parameter Values That Matter

**Shared geometry:**
| Parameter | Value | Why |
|---|---|---|
| SOD | 500mm | Source-to-object distance, nominal scan geometry |
| ODD | 200mm | Object-to-detector distance |
| Detector | 512×512, 0.2mm pitch | Industrial/medical flat-panel class |
| Random seed | 42 | All runs bit-identical for reproducibility |
| sigma_s nominal | 2mm (Navy), 3mm (NIH) | Open-config freehand position uncertainty |
| sigma_theta nominal | 1deg (Navy), 1.5deg (NIH) | Open-config orientation uncertainty |
| sigma_s stress | 10mm (Navy) | Stress test condition |
| sigma_theta stress | 5deg (Navy) | Stress test condition |

**Navy-specific:**
| Parameter | Value | Why |
|---|---|---|
| mu_steel | 0.1149 mm⁻¹ | NIST XCOM, iron at 200keV (5.7% transmission through 25mm) |
| mu_lead | 1.133 mm⁻¹ | Fiducial markers at 200keV |
| Voxel size | 0.1mm isotropic | Sub-crack resolution needed |
| Grid | 250×250×250 | 25mm phantom at 0.1mm pitch |
| Primary crack target | 0.8mm | ASTM E1742 compliance (Rose criterion CNR ≥ 4) |
| N_shots primary | 100 | Also run 20, 50, 200 for accuracy vs shot count figure |

**NIH-specific:**
| Parameter | Value | Why |
|---|---|---|
| mu_bone | 0.048 mm⁻¹ | Cortical bone at 70keV |
| mu_brain | 0.021 mm⁻¹ | Soft tissue at 70keV (~40 HU) |
| mu_blood | 0.023 mm⁻¹ | Fresh blood at 70keV (DeltaHU = 40–50) |
| mu_BaSO4 | 0.31 mm⁻¹ | Fiducial markers at 70keV |
| Voxel size | 1.0mm isotropic | Head CT resolution |
| Grid | ~200×160×140 | 90×70×65mm ellipsoid skull |
| Primary hemorrhage | 5mm sphere | Aim 2 Go/No-Go target |
| Shot arc | Restricted 180° | ICU bedside access constraint |
| N_shots | 80 | ICU scan time < 15min |

**SDSG solver:**
| Parameter | Value |
|---|---|
| Optimizer | scipy.optimize.least_squares, method='lm' |
| xtol / ftol | 1e-10 |
| max_nfev | 200 |
| x_scale | 'jac' |
| Initial guess | source=[0,0,-SOD], detector=[0,0,ODD], Euler=[0,0,0] |
| Failed U7 threshold | final cost > 2.0px → exclude from merge |

**CRITICAL implementation notes:**
- Use ASTRA `cone_vec` geometry type exclusively — not standard `cone`
- SSIM and PSNR: always pass `data_range` explicitly — default silently halves SSIM for float arrays
- Sinogram axis order: `(det_rows, N_projections, det_cols)` — document this, common bug source
- mART: clamp at epsilon=1e-6 minimum to prevent log(0)
- Quaternion averaging for rotations — NOT naive Euler averaging (gimbal lock)
- Delete all ASTRA data3d and algorithm objects after each reconstruction (CUDA memory leak)
- Do NOT mix 200keV (Navy) and 70keV (NIH) attenuation values

---

## Code Standards

### General

- Write production-quality code, not demo code
- Handle edge cases and errors properly
- Think about performance and scalability from the start
- Add comments only for non-obvious logic — not basic syntax
- Components: `PascalCase` | Files: `snake_case` (Python convention) | Constants: `UPPER_SNAKE_CASE`

### API Routes

- All responses return consistent shape: `{ data, error, status }`
- Always validate input with zod before any processing
- Never expose internal error messages or stack traces to client
- All routes require authentication unless explicitly marked public
- Use proper HTTP status codes

### Testing

- Write tests alongside every new feature — not after
- Unit tests for utility functions and business logic
- Integration tests for API routes
- Every PR must have passing tests before merge

### Git

- Commit after every completed step
- Commit messages: `type: description`
  - `feat:` new feature | `fix:` bug fix | `refactor:` no behavior change
  - `test:` adding tests | `docs:` documentation only
- Never commit `.env` files
- Branch naming: `feature/name`, `fix/name`, `refactor/name`

---

## Security Rules — Never Violate These

- Never hardcode API keys, secrets, or credentials anywhere
- Never commit `.env` or `.env.local` files
- Never expose service role keys to the client
- Always use environment variables for sensitive config
- Always sanitize user input before database operations
- RLS is not optional — every table gets policies

---

## How I Want You to Behave

### Autonomy

- Work through multi-step tasks without asking for permission at every step
- Make reasonable technical decisions and document them in PLAN.md
- If genuinely blocked, document the blocker and move to the next step
- Don't ask clarifying questions when the answer is obvious from context

### Communication

- Be direct and concise — I'm on a phone screen
- When starting a task, give me a brief plan before executing
- When finishing, give me a brief summary of what was done
- Flag important decisions you made so I can review them
- Never give me walls of explanation — action first, explanation if needed

### When You're Unsure

- Make the most reasonable decision based on context
- Document the decision in a code comment or PLAN.md
- Flag it clearly: `# DECISION: chose X over Y because Z`
- Keep moving — don't stop the whole task over one uncertainty

---

## Skills System

Skill files live in `.claude/skills/`. Load them explicitly when needed:

| Task Type | Load This Skill |
|-----------|-----------------|
| Frontend UI work | `.claude/skills/frontend.md` |
| Backend API work | `.claude/skills/backend.md` |
| Database design | `.claude/skills/database.md` |
| Marketing copy | `.claude/skills/marketing.md` |
| Technical writing | `.claude/skills/technical-writing.md` |
| Embedded/firmware | `.claude/skills/embedded.md` |

**Auto-load rules:**
- React/Next.js component work → read frontend skill first
- API route or server logic → read backend skill first
- Schema design or migration → read database skill first
- Landing page or copy → read marketing skill first

---

## PAQ Constraints — Important

- No work on DoD SBIR applications or anything that creates a conflict of interest with Air Force work
- GrantPilot targets civilian federal grants (FEMA, HUD, EPA, DOT) and municipal clients only — explicitly cleared
- Written approval from base legal required before shipping any code commercially
- When in doubt about a conflict, flag it and wait for my guidance

---

## Quick Reference — Things I Always Want

1. PLAN.md created and maintained on every task
2. Git commit after every completed step
3. `progress.txt` updated after each step
4. Tests written alongside features
5. No secrets in code ever
6. Production quality, not demo quality
7. Resume from PLAN.md without re-explanation
8. Always use `conda run -n sdsg_sim python` — never system Python
