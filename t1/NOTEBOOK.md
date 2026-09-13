# uot / t1 — Lab notebook

Project: **Do language models have an internal algebra of physical units?** (attempt t1)
Researcher: Claude (autonomous). PI: Charlie (async).

## FLAGS (PI must see; none block work)

- **F1 (2026-09-12) Gated models — RESOLVED same day.** Gemma-2-9B / Gemma Scope were 403 at audit; the PI accepted the Gemma license within the hour and access now works. Third family = Gemma-2-9B as the brief specifies. `meta-llama/Llama-3.1-8B` remains gated and is not needed. Mistral-7B-v0.3 kept as an optional fourth family for Phase 3 scaling/robustness (DR-002).
- **F2 (2026-09-12) No infini-gram index for OLMo-3's corpus (Dolma 3).** Only `v4_dolma-v1_7_llama` (OLMo-1.7 pretraining corpus) is served. Assumption adopted: Dolma-1.7 counts are the corpus-frequency instrument, treated as a *proxy* for OLMo-3's corpus; all frequency claims are worded accordingly (DR-002).
- **F4 (2026-09-13 00:30) GPUs on rosetta4 are now fully occupied** (another user's 8-GPU job at ~98% util on every device since ~23:50). Per the shared-cluster rule I am not launching GPU work here while that holds. `ssh rosetta11`/`rosetta5` fails non-interactively: the system ssh_config routes through `jump.csail.mit.edu` (needs 2FA), and a direct connection refuses my fresh key because sshd cannot read `~/.ssh/authorized_keys` in AFS (ACL is owner-only). I did NOT change any AFS ACL. **Ask:** either (a) grant me a way onto rosetta11 (e.g. an ssh key whose `authorized_keys` sshd can read, or tmux session there), or (b) say it is fine to share rosetta4 GPUs with the running job. Until then: CPU-side work continues; GPU polling every ~20 min.
- **F5 (2026-09-13) conda activation takes ~50 s on this NFS under load** and Python imports of numpy/pint take 20–100 s when the node is loaded. All scripts are invoked with the env's python binary directly (`t1/env/bin/python`), never via `conda activate`.
- **F3 (2026-09-12) Node driver.** rosetta4 has driver 550 (CUDA 12.4), so torch must be a cu124 build (README in home says 560 — that was a different node). Env uses torch 2.6.0+cu124.

## Conventions

- One conda env at `t1/env` (see `env_setup/create_env.sh`). Activate with `conda activate /data/rbg/users/charlie/interp/uot/t1/env`.
- Code lives in `src/uot` (library) and `scripts/phase{0,1,2,3}/` (thin, numbered entry points). No ad-hoc scripts in the root.
- Every experiment has an ID `E<phase>.<n>` (e.g. `E0.3`), a config under `configs/`, outputs under `runs/E0.3/`, and an entry here.
- Compute ledger at the bottom of this file.

---

## 2026-09-12 — Day 0: reading, environment audit, design

### Environment audit
- Node rosetta4: 8× A100-PCIE-40GB, all idle at audit time; 96 CPUs; 503 GB RAM; NFS home `/data/rbg/users/charlie` with 9.9 TB free. rosetta11 has 4 idle A6000-48GB; rosetta5 has 7 idle A6000s. Cluster is shared: usage checked at `http://rosetta1.csail.mit.edu:4444/` before every launch (a helper script will do this).
- Internet: HF hub and arXiv reachable. HF token present (valid, `whoami` OK) but Gemma/Llama gated (F1).
- infini-gram API reachable; indices verified: `v4_dolma-v1_7_llama` (count("meters per second")=242,351), `v4_piletrain_llama`, `v4_c4train_llama`, `v4_rpj_llama_s4`. No OLMo-2/OLMo-3/Dolma-3 index (F2).
- Env build (`env_setup/create_env.sh`) started 23:50: python 3.11, torch 2.6.0+cu124, transformers 4.x, nnsight, pyvene, pint, sklearn, etc. Library ↔ model support verified in a smoke test once built (see E0.0 below).

### Reading synthesis: the rival paper's DAS setup (arXiv 2606.03982) and how we mirror + extend it

**What they did.** Sasaki et al. study two-quantity *comparison* ("Which is larger, 110 cm or 1.2 m? The answer is"). Their pipeline is behavioural → surrogate → causal:
1. *Behavioural*: accuracy vs. log-space Quantity Margin QM = (log r1 − log r2) + (log s(u1) − log s(u2)); accuracy collapses near QM≈0; errors concentrate where the numeral cue and the unit-scale cue disagree. 5 unit settings (metric/imperial length, metric mass, mixed), values 1e-3 ≤ r < 1e4, decimal notation with 3 fractional digits. Models: Qwen3-4B/8B/14B-Base, Olmo-3-1025-7B, Olmo-3-1125-32B; instruct Qwen3-4B/8B. Greedy decoding, exact match; multiple prompt templates (postposed/preposed × larger/smaller × short/long unit form).
2. *Surrogate*: ridge regression from cue features to the LM's log-prob margin m_LM = mean-token-logprob(q1 string) − mean-token-logprob(q2 string). Feature families: QM alone; global quantities (log r + log s per side); primitive components; NumLogDiff & UnitLogDiff (continuous, signed, thresholded); all heuristics. Signed/thresholded NumLogDiff+UnitLogDiff explain the most residual variance (R²_partial ≈ 0.44–0.47 overall, ≈ 0.70 near the boundary); global (shared-scale) quantities explain ≈ 0.
3. *Causal (DAS)*: orthogonal rotation R ∈ R^{d×d} learned at ONE residual-stream site (layer ℓ, token position p); rotated space split into k variable subspaces + residual; distributed interchange intervention swaps subspace j from source s_j into base b; trained by CE against the high-level counterfactual label y_cf = 1[NumLogDiff(s1)+UnitLogDiff(s2) > 0]. Total intervention dim fixed at 1024 split evenly (512/variable; sweeps 256/512/1024/2048). Token sweep shows the effect concentrates at the *last token of u2*; layers 8–24 step 2; 20K train / 1K eval interventions; output-changing and output-preserving interventions balanced so chance IIA = 0.5; 3 seeds; 10K-bootstrap CIs. Compared variable sets: {NumLogDiff, UnitLogDiff} (best, IIA≈0.93–0.95 at layers 16–20 on Qwen3-4B-Base) vs. {GlobalLogX, GlobalLogY} vs. identity {log r1, log s1, log r2, log s2} vs. {NumLogDiff, log s1, log s2}. All baselines are above chance but consistently lower.

**Their limitations (their words):** single-step two-quantity comparison only; linear surrogates; DAS finds subspaces, not circuits; not extended to reasoning models / CoT. **Not touched at all:** the *dimension* variable — every comparison is within one dimension, so dimension is constant and invisible to their design.

**What is missing by our standards (gaps we fill):**
- No matched-rank *random-subspace* baseline and no "dormant pathway" checks (Makelov, Lange & Nanda 2023): high IIA from a 512-dim learned rotation at a late layer is exactly the regime where the illusion lives.
- Only denoising-style interchange; no noising/necessity direction, no dose-response, no check that the *other* variables (e.g. numeral value) survive the swap.
- Single prompt for the causal analysis; no template distribution in the DAS stage.

**How our design mirrors it.** Same model suite core (Qwen3-4B/8B/14B-Base, Olmo-3-7B/32B; Qwen3-4B/8B instruct), same readout for base models (mean token log-prob over candidate answer strings; greedy exact match as secondary), same DAS machinery (pyvene rotated-space interchange; total intervention dim swept over {64,128,256,512,1024} split evenly across variables; IIA with balanced counterfactuals, chance 0.5; layer × position sweeps; 3 seeds; bootstrap CIs over prompts), same "rival variable sets compared head-to-head at matched rank" logic.

**How it extends it.** (i) The high-level variable is the *dimension* d(q) ∈ ℤ^k, not the scale; our tasks vary dimension and hold scale/value nuisance. (ii) Two rival causal models are aligned at matched rank: M_alg (a lattice-valued D per quantity; licensing = equality test; composition = vector addition) vs. M_heur (per-unit lexical-identity variables + a pairwise-compatibility cue that need not respect the group structure). (iii) Full control regime: matched-rank random subspaces, rank sweeps, denoising *and* noising, dose–response along the subspace, value-readout preservation after a dimension swap, binding-contamination checks in multi-quantity prompts, and a text-rewrite baseline. (iv) Geometry phase before causality: probes that must extrapolate to held-out lattice points, which a cue-set cannot do. (v) Composition (H4): intervening on an input's dimension must propagate lawfully to a derived quantity's dimension, including mid-CoT.

### Other required reading — what each contributes to our design
- **Kantamneni & Tegmark (2502.00873)**: parametric-fit → patch-the-fit → component-level modelling. Our analogue: fit rep(q) ≈ W·e(q) + b (e = exponent vector), then *patch the fitted reconstruction* into the residual stream and measure logit-difference recovery vs. a matched-dim PCA baseline and vs. full-layer patching (Phase 1 secondary; makes "linear in the exponent vector" a causal claim, not a probe claim).
- **Nanda et al. Othello (2309.00941)**: parameterisation flips verdicts (BLACK/WHITE nonlinear vs MINE/YOURS linear). Our analogue: before concluding "not linear" we test alternative parameterisations of the dimension variable — signed exponents vs. one-hot-per-(axis,exponent) vs. |exponent| + sign vs. "named-dimension categorical" — and pick by held-out fit *before* the lattice-holdout headline (pre-registered in PREDICTIONS_phase1).
- **Feng & Steinhardt (2310.17191)**: binding IDs are additive vectors attached to entity/attribute tokens; contamination control: in two-quantity prompts, a "dimension swap" on quantity 1 must not change the model's readout of quantity 2's dimension or value, and position-swap controls separate binding-ID effects from dimension effects. Also motivates the anaphor position class ("that speed"): dimension must be retrievable from the referent, not the unit lexeme.
- **Park, Choe, Jiang & Veitch (2406.01506)**: an algebraic structure (hierarchy) → geometry (orthogonality, polytopes) paper with the right control (shuffled unembeddings, random-parent baselines, train/test split of the defining token sets). Our analogue: the free-abelian-group structure → linear (additive) geometry; our controls: shuffled-lexeme→dimension assignments (Hewitt–Liang control task), random-triple baselines for vector-arithmetic identities, and cross-lexeme train/test splits.
- **Zhang & Nanda (2309.16042)** / **Heimersheim & Nanda (2404.15255)**: symmetric token replacement (no Gaussian noise), logit difference for localisation, KL/accuracy for distribution claims; denoising (sufficiency) vs. noising (necessity) both reported; single-layer first, sliding window only if effects are diffuse.
- **Geiger et al. DAS (2303.02536) / causal-abstraction survey (2410.20161)**: interchange-intervention accuracy as the alignment metric; the high-level causal model must be fully specified (variables, mechanisms) so that counterfactual labels are computed by the model, not by hand.
- **Makelov, Lange & Nanda (2311.17030)**: a DAS subspace can steer outputs via a *dormant* direction disconnected from the model's actual computation. Their success-case criteria we adopt: (1) the found subspace must separate the natural conditions when *natural* activations are projected onto it (no dormant directions); (2) matched-rank random-subspace and full-component patching comparisons; (3) a probe/mean-difference-derived subspace should give comparable IIA to the DAS-found one; (4) necessity: projecting the subspace out on natural inputs should hurt the task.
- **TypeProbe (2607.08339)**: nearest neighbour; linear probes for types in code models, selectivity = task − control accuracy, cross-language transfer, adversarial renaming. No causal step, no lattice structure. We reuse the selectivity metric and the adversarial-lexeme idea (units whose surface form suggests the wrong dimension, e.g. invented unit names that resemble real ones).
- **Nikankin et al. (2410.21272)**: bag-of-heuristics = many sparse neuron-level cue detectors; the *extrapolated null* for us is a set of unit-pair / unit-identity cues without group structure (A2). Their neuron-level analysis is out of scope for t1 except as a Phase-3 stretch.

### Design decisions today
See `DESIGN_phase0.md`, `decisions/DR-001-*.md` (adversarial pass on the design), `decisions/DR-002-*.md` (model suite + frequency instrument deviations), `PREDICTIONS_phase0.md` (committed before any battery run), `QUESTIONS.md`.

---

## Compute ledger

| Phase | GPU-hours (approx) | Notes |
|---|---|---|
| 0 | 0 | env build only so far |
