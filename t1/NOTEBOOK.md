# uot / t1 — Lab notebook

Project: **Do language models have an internal algebra of physical units?** (attempt t1)
Researcher: Claude (autonomous). PI: Charlie (async).

## FLAGS (PI must see; none block work)

- **F1 (2026-09-12) Gated models — RESOLVED same day.** Gemma-2-9B / Gemma Scope were 403 at audit; the PI accepted the Gemma license within the hour and access now works. Third family = Gemma-2-9B as the brief specifies. `meta-llama/Llama-3.1-8B` remains gated and is not needed. Mistral-7B-v0.3 kept as an optional fourth family for Phase 3 scaling/robustness (DR-002).
- **F2 (2026-09-12) No infini-gram index for OLMo-3's corpus (Dolma 3).** Only `v4_dolma-v1_7_llama` (OLMo-1.7 pretraining corpus) is served. Assumption adopted: Dolma-1.7 counts are the corpus-frequency instrument, treated as a *proxy* for OLMo-3's corpus; all frequency claims are worded accordingly (DR-002).
- **F4 — RESOLVED 01:45: PI enabled ssh to rosetta11/rosetta18; Phase-0 GPU work runs on rosetta18 (7 idle A6000-48GB, driver 550). Original flag:** GPUs on rosetta4 are now fully occupied (another user's 8-GPU job at ~98% util on every device since ~23:50). Per the shared-cluster rule I am not launching GPU work here while that holds. `ssh rosetta11`/`rosetta5` fails non-interactively: the system ssh_config routes through `jump.csail.mit.edu` (needs 2FA), and a direct connection refuses my fresh key because sshd cannot read `~/.ssh/authorized_keys` in AFS (ACL is owner-only). I did NOT change any AFS ACL. **Ask:** either (a) grant me a way onto rosetta11 (e.g. an ssh key whose `authorized_keys` sshd can read, or tmux session there), or (b) say it is fine to share rosetta4 GPUs with the running job. Until then: CPU-side work continues; GPU polling every ~20 min.
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

## 2026-09-13 — Day 1 (early): library built, E0.0 audits, infrastructure friction

### E0.0 — tokenisation audit (`runs/E0.0/tokenization_audit.json`)
- Invented lexemes (CVC/CVCC, 4–6 chars) tokenise to 2 tokens for ~80%, 3 for ~17%, 1 for ~3% (Qwen3 and OLMo-3 tokenisers behave identically on all audited strings; Gemma-2: 85/11/4%). No lexeme exceeds 3 tokens → no tokenisation pathology (PREDICTIONS_phase0 "distrust a FAIL" clause 1 is satisfied).
- Real units: short symbols are 1 token for 65/79 (Qwen3/OLMo-3) and 72/79 (Gemma-2); long plurals 1–4 tokens.
- Compound renderings split sensibly, e.g. `kg·m/s²` → [kg][·][m][/s][²]; `kilogram meters per second squared` → 6 tokens. The mean-per-token log-prob readout is therefore comparing candidates of 3–8 tokens; length-matched distractors (see below) keep this fair.

### Library ↔ model support (E0.0)
- transformers 4.57.6 has `Qwen3ForCausalLM`, `Olmo3ForCausalLM`, `Gemma2ForCausalLM`.
- pyvene 0.1.8 ships module maps for Qwen2/Olmo(v1)/Gemma2/Llama/Mistral but **not Qwen3 or Olmo3**. nnsight 0.7 wraps any HF model.
- **Decision (engineering default changed):** implement DAS ourselves in plain PyTorch (forward hooks on the residual stream at chosen layer/positions; orthogonal rotation via `torch.nn.utils.parametrizations.orthogonal`; interchange = swap the first k rotated coordinates). Reasons: (i) uniform treatment of the control regime (matched-rank random subspaces, rank sweeps, probe-derived subspaces, noising/denoising) without fighting a library's abstractions; (ii) no dependence on pyvene's per-architecture maps; (iii) ~200 lines, fully auditable. pyvene remains available for a cross-check on Gemma-2 (supported) in Phase 2.

### Stimulus library status (E0.1)
- 17 unit tests green (dimension group laws; every registry unit's dimension and SI scale checked against Pint; near-miss pairs; rendering; lexeme generator; task generators incl. yes/no balance and both surface orders per relation; Phase-1 mention spans).
- Two design fixes after inspecting generated items (logged in DR-001 addendum below): (1) distractors are now **length-matched** — every distractor uses all the lexemes of the correct answer (inverted / exponent-permuted / random lattice point over the same lexemes), replacing the earlier "single unit" distractor that made the correct answer the longest candidate in 62% of items; (2) compound units are composed **system-consistently** (metric or imperial, never "foot kilograms") and T1 inputs use natural everyday units (no "126 yd in 9 ms").
- Built: `data/phase0/{T1..T5}_s0.jsonl` (240 items per condition; T1 has 4 conditions + 2 twin sets = 1440 items) and `T1_s1.jsonl` (second stimulus seed for the headline cell).

### Infrastructure friction (all logged as flags F4/F5)
- All 8 A100s on rosetta4 became occupied by other users' jobs at ~23:50; no GPU work launched since.
- The NFS home is the bottleneck for everything: `conda activate` ≈ 50 s, `import torch` 20–100 s, and python processes sit in `rpc_wait_bit_killable`. My own 64 GB OLMo-3-32B download was a contributor; it is paused (SIGSTOP) until GPU work is possible anyway.
- infini-gram API occasionally times out; the frequency cache now records misses and re-queries on the next run.

### Anomaly A1 (2026-09-13 01:20) — "invented" lexemes that are real words
The frequency table (E0.4) showed 34/480 invented-unit answer strings with nonzero Dolma-1.7 counts, e.g. "port/h" (634), "kind per km", "mick cm": the CVC generator's English filter relied on `/usr/share/dict/words`, which does not exist on this node, so real words (port, kind, mick, …) leaked through. **Fix, before any headline run:** (1) `data/wordlists/words_alpha.txt` (dwyl English words, 370k entries) is now the filter; (2) new script `00_verify_lexemes.py` queries Dolma-1.7 for every candidate lexeme (singular and plural) and keeps only those with count ≤ 50, writing `data/lexemes/verified.json`; `LexemePool` draws from this verified pool. This turns "zero corpus support by construction" into "verified ≤ 50 Dolma-1.7 occurrences per lexeme", which is a stronger, auditable statement. Stimuli will be rebuilt from the verified pool; the E0.2 CPU debug run (on the pre-fix stimuli) is kept as a pipeline check only and will not enter the gate report.

### E0.3 first look (2026-09-13 02:05) — Qwen3-4B-Base, T1_s0 (observation only, no interpretation yet)
acc_cand by condition: FAM-NAMED 0.92, FAM-UNNAMED 0.88, INV-LEX 0.92, INV-BASE 0.95, twins 0.87/0.87 (n=240 each, chance 0.25). Order-swapped items are *not* worse (INV-LEX swapped 0.96 vs in-order 0.88). This is far above PREDICTIONS_phase0 (P0.3 predicted 0.60 for INV-LEX). Wrong answers are mostly `sign_flip` and `exp_swapped` distractors. acc_gen shows 0.00 — a matcher bug (generation "square chaurp. What is…" was rejected because of the period); fixed, recomputed at analysis time from stored generations. Before interpreting: run prior-only controls (E0.3b: no-context and no-units variants) to bound how much of this is candidate-prior + relation wording rather than composition over the input lexemes.

### Generation readout, first look (Qwen3-4B-Base, T1_s0 INV-LEX; 2026-09-13 09:20)
Free greedy generation after the prompt (which ends in the output *number*) is a noisy readout: 80/240 generations contain no unit at all (the model either continues the digits of the number, e.g. "0.004458" → "529032258064", or closes the sentence and starts a new question), 109 are dimensionally correct, 51 are dimensionally wrong. The wrong ones are systematically *simplified* compositions: exponents dropped or a factor omitted ("snoorp per shoath" for a force whose correct unit is "vraunt snoorp per shoath²"; "skeesp" for a volume "skeesp³"). Two consequences: (i) the candidate readout stays primary (as pre-registered), reported alongside `acc_gen_dim|parsed` (dimension-correct among generations that contain a unit); (ii) exponent handling is the weak spot of free composition — worth a dedicated Phase-1 look (is the exponent *magnitude* represented, or only which lexemes participate and on which side of the slash?). Also implemented: a dimension-aware parser (`uot/parse_units.py`) so any dimensionally-correct surface form ("N", "kg-cm", "wrults cubed") counts.

### E0.3b — prior-only controls, Qwen3-4B-Base, T1_s0 (2026-09-13 09:30)
| condition | full prompt | no units in scenario (relation + definitions kept) | no context (definitions + "The unit is") | chance |
|---|---|---|---|---|
| FAM-NAMED | 0.92 | 0.59 | 0.16 | 0.25 |
| FAM-UNNAMED | 0.88 | 0.42 | 0.08 | 0.25 |
| INV-LEX | 0.92 | 0.70 | 0.44 | 0.25 |
| INV-BASE | 0.95 | 0.76 | 0.31 | 0.25 |
| INV-LEX-TWIN | 0.87 | 0.50 | 0.14 | 0.25 |
| INV-BASE-TWIN | 0.87 | 0.41 | 0.03 | 0.25 |

**Observation.** Candidate prior alone is at or below chance for real units (the composed correct string is not favoured a priori). Removing the units from the scenario but keeping the relation sentence (and, for INV, the lexeme definitions) retains 0.42–0.76: the model can select the right lattice point from the *verbal* relation ("mass squared times length" → "kg² mi") plus the candidate strings, without any slot binding. The full prompt adds +0.19 to +0.46 on top.
**Interpretation (suggestive).** Two routes contribute to T1 candidate accuracy: (R1) relation-words → exponent pattern over the available lexemes (a verbal/A1-style route that never needs a dimension variable attached to the mentioned quantities), and (R2) binding the mentioned quantities' units to slots and composing. The G0 criterion is about competence, and it passes regardless; but the red-team appendix must carry this: Phase-1/2 must probe the *quantities'* dimension representations (positions at/after the unit tokens and at the pre-answer token), and the causal tests must swap dimension at the quantity, not the relation words. A cleaner behavioural cell for R2 alone is also worth adding: relation *not* verbalised and *not* named (e.g. "Q = 12 blorks × 3 zims" style definitions by example) — deferred to Phase 1 (E1.0) rather than reopening Phase 0.
**What would change my mind:** if the full-prompt increment over "no units" vanished on order-swapped items (it does not: swapped ≥ in-order), R2 would be absent.

### E0.3 — T2–T5 across four base models (2026-09-13 10:00; observation)
- **T3 formula selection** (4-way, chance 0.25): Qwen3-4B/8B 0.91/0.97 (named), 0.74/0.76 (unnamed), 0.83/0.90 (invented); OLMo-3-7B 0.77/0.58/0.60; Gemma-2-9B 0.80/0.53/0.60. Unnamed lattice points are harder than named ones for every model (−0.15 to −0.28).
- **T2 consistency** (yes/no): 0.50–0.76 everywhere; Gemma at chance in every cell; wide template CIs. **T4 conversion**: Qwen3-8B 0.78–0.92, Qwen3-4B 0.70–0.85, OLMo-7B 0.68–0.81, Gemma at chance. **T5 error spotting** (3-way, chance 0.33): Qwen ≈0.45–0.56, OLMo/Gemma at chance.
- Caveat before interpreting T2/T4/T5: a base model's " yes" vs " no" log-prob comparison after "Answer:" can be dominated by a constant bias, which yields exactly chance under balanced labels even if the margin *ranks* items correctly. Added bias-free readouts (AUROC of the margin; median-calibrated accuracy) — reported next. Instruct models (queued) are the intended readout for judgment tasks anyway; base and instruct stay separate.
- Anomaly A2: in T1, accuracy is *lowest* at lattice distance 2 (speed/area/L·T) and for the *most frequent* answer strings (log-count > 4.5: km/h, m/s, m²). To inspect: which distractor wins there (sign-flip "km·h"?), and whether the mean-per-token readout penalises short familiar strings.

### Calibrated yes/no readouts (E0.3, base models; 2026-09-13 10:20)
Raw accuracy on T2/T4 hides a constant answer bias: Gemma-2-9B answers "yes" on 75–100% of items, Qwen3/OLMo-3 answer "no" on 70–99%. Threshold-free readouts tell a different story (AUROC of the yes-minus-no margin vs label):
- T2 consistency: INV-LEX AUROC 0.89 (Qwen3-4B) / 0.96 (Qwen3-8B) / 0.97 (OLMo-7B) / 0.70 (Gemma); INV-BASE 0.85 / 0.82 / 0.94 / 0.66; familiar pairs 0.64–0.84 (Qwen/OLMo), ≈0.5 (Gemma).
- T4 conversion: 0.80–0.98 (Qwen/OLMo), 0.72–0.77 (Gemma).
Interpretation (suggestive): base models *rank* dimensional consistency well but do not map it to a yes/no token without calibration; invented units (dimension stated in the definition) are easier than familiar near-miss pairs (dimension must be recalled). Readout policy from here: for two-way judgment tasks report AUROC and median-calibrated accuracy alongside raw accuracy; P0.6/P0.7 are scored on raw accuracy as pre-registered.

### Anomaly A2 resolved → stimulus fix (T1 v2)
The distance-2 / high-frequency dip was entirely the **area** relation (acc 0.36–0.39 vs ≥0.79 elsewhere): its exponent-dropped distractor " m" is a strict *prefix* of the correct " m²", and per-token log-prob readouts favour the prefix. 37/1440 T1_s0 items had prefix-related candidate pairs. Fix: candidate sets now exclude any distractor rendering that is a prefix of (or prefixed by) another candidate; rebuilt as `T1v2_s{0,1}.jsonl` (versioned so the finished runs stay reproducible). T1 will be re-run on all models with v2; v1 results are kept and reported in the gate appendix. This is a stimulus-validity fix; no threshold changes.

### Anomaly A5 (2026-09-13 10:50) — letter-position bias in T5 (and mildly T3)
Base models almost never choose option B in T5 (OLMo-7B picks B 27/720, Gemma 10/720); accuracy conditional on the correct position is 0.03–0.37 for B vs 0.44–0.85 for C. T3 (four options) shows a milder D-avoidance in the 4B/7B/9B models and none in Qwen3-14B. Readout fix (no stimulus change): per-position median calibration (subtract each option position's median score across items, then argmax), reported as `acc_calibrated` for T3/T5. The instruct model shows the same B-avoidance, so this is a format prior, not a base-model artefact only.
