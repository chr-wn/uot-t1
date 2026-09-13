# DESIGN — Phase 1: geometry of the dimension variable

Status: drafted 2026-09-13 while the last Phase-0 runs finish; finalised with the G0 decision record. No Phase-1 model run precedes `PREDICTIONS_phase1.md`.

## 0. What Phase 0 tells Phase 1
All base models compose unit expressions for invented lexemes and unnamed lattice points at ≈0.85–0.95 (candidate readout) with no surface-order penalty, but ~50–75% of that accuracy is available from the relation words + candidate structure alone (E0.3b). So the behavioural result licenses the search for a dimension variable but does not locate it; Phase 1 must ask where, if anywhere, the *mentioned quantities'* dimension is represented, whether that representation is linear in the exponent vector, and whether a composed dimension exists before the answer is produced.

## 1. Stimuli (`uot.tasks.p1_mentions`, `data/phase1/`)
Single-quantity passages with an anaphoric continuation; character spans → token positions:
`value` (last numeral token), `unit` (last unit token), `mention_end` (the period after the mention), `anaphor` (the referring noun in the next sentence), `last` (final token).
Conditions: REAL-BASE (7 SI base dimensions, many lexemes each), REAL-NAMED (all named-derived units: N, J, W, Pa, Hz, V, Ω, …), REAL-LATTICE (all 124 points with |e_L|,|e_M|,|e_T| ≤ 2 composed from natural units, named and unnamed), DIMLESS (bare numbers, "percent", "radians"), INV-LEX (invented lexemes defined in context), XLING (es/de/fr long forms in English sentences).
Noun families: *neutral* (noun carries no dimension: "the reading was 12 meters"), *revealing* ("the length was 12 meters"), *noun_only* ("the length was 12", no unit) — the last two separate "dimension of the quantity" from "identity of the unit lexeme".
Additional position class from Phase 0: the **pre-answer** token of T1v2 prompts (the output number, where a *composed* dimension must be represented if the model composes); probes trained on mentions are applied there (transfer, no retraining) and also trained there directly.
Sizes: ~400 items per condition × 3 families → ~6,000 mention items per seed; 2 seeds.

## 2. Caching (`uot.cache`, `scripts/phase1/02_cache_residuals.py`)
Residual stream after every second layer (+ last), at the five positions, fp16. ≈ 0.5 MB/item for Qwen3-4B → ≈ 3 GB per model per seed. Models: Qwen3-4B-Base (debug + full), Qwen3-8B/14B-Base, OLMo-3-7B, Gemma-2-9B; OLMo-3-32B only for the headline cell if time allows.

## 3. Probes (`uot.probes`)
Primary: ridge regression residual → exponent vector over (L, M, T) [7 axes for REAL-BASE], evaluated by nearest-lattice-point accuracy (among the label set) and per-axis R².
Splits: random (upper bound), **cross-lexeme** (group folds by unit lexeme; probe never sees a test lexeme), **lattice holdout** (whole exponent combinations held out; the 3 base points always in train), cross-family (train neutral → test revealing / noun_only), cross-condition (train REAL → test INV-LEX and XLING).
Controls: Hewitt–Liang control task (lexeme → random lattice point, consistent per lexeme) under the same splits; selectivity = task − control. Matched-capacity: same ridge alpha, same dimensionality.
Parameterisation check (Othello lesson, pre-committed order): (a) linear-in-exponent; (b) sign/magnitude split (|e| and sign(e) as separate targets); (c) categorical (multinomial logistic over lattice points) — in-distribution only; (d) 1-hidden-layer MLP with the same input. Verdicts are drawn from (a) vs (c)/(d) on random splits *and* from (a) on lattice holdout; if (d) ≫ (a) on random splits but (a) extrapolates, the linear component is what generalises.
Geometry: centroid per lattice point at the best (layer, position); additivity residual ‖c(d1·d2) − c(d1) − c(d2) + c(1)‖ relative to a random-triple baseline; dimensionless-at-origin: projection of DIMLESS items onto the probe → distance from 0.
Causal check of the parametric fit (Kantamneni & Tegmark style, cheap): patch the probe-reconstructed exponent-subspace of a source mention into a base mention (T2-style yes/no prompt) and measure logit-difference recovery vs. matched-dimension PCA baseline — a preview of Phase 2, run only at the best site.
Gemma Scope (Gemma-2-9B, layer ≈20 residual SAE, 16k): correlate latent activations with exponent axes; report whether a small set of latents spans the probe direction (secondary, descriptive).

## 4. Analyses and figures
Layer × position heatmaps of selectivity (task − control) per split; lattice-holdout accuracy per held-out point (which points extrapolate: named vs unnamed, distance); additivity-residual histogram vs baseline; transfer matrix (train condition × test condition); pre-answer-position transfer from mentions.

## 5. G1 criterion — see PREDICTIONS_phase1.md (written before any Phase-1 headline run)

## 6. Experiment list
| ID | What |
|---|---|
| E1.0 | build mention stimuli (2 seeds), tokenisation/position audit |
| E1.1 | cache Qwen3-4B-Base; full probe suite (debug) |
| E1.2 | cache + probes for 8B/14B/OLMo-7B/Gemma-9B |
| E1.3 | parameterisation comparison; additivity; dimensionless-at-origin |
| E1.4 | pre-answer-position transfer (T1v2 prompts) |
| E1.5 | probe-reconstruction patching (causal preview) |
| E1.6 | Gemma Scope latent reconnaissance |
| G1 | gate report, red-team appendix, decision record |
