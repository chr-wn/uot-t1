# G1 gate report — Phase 1 geometry (2026-09-13)

Companions: `reports/G1_red_team.md`, `decisions/DR-004-G1-decision.md`, `PREDICTIONS_phase1.md`, `figures/phase1/`, `runs/E1.*`.

## 1. What ran
- **Stimuli.** Single-quantity mention passages with an anaphoric continuation (`uot.tasks.p1_mentions`); positions value / unit / mention_end / anaphor / last. Two sets: P1 (5,916 items/seed: base, named, 124 compound lattice points ×4 lexeme variants, dimensionless, invented, cross-lingual) and **P1b** (5,700 items/seed; registry densified with 43 extra single-word named units so every derived point has ≥3–12 lexemes). Two seeds each. Also T1v2 and arithmetic prompts cached at value/unit/pre-answer positions (Qwen3-4B).
- **Models.** Qwen3-4B/8B/14B-Base, OLMo-3-7B, Gemma-2-9B; residual stream every 2nd layer, fp16 (`cache/`, 2.7–6 GB per file).
- **Probes.** PCA-512 → ridge (α by LOO-CV) to the (L, M, T) exponent vector; multinomial logistic over lattice points (categorical); sign/magnitude split; 1-hidden-layer MLP. Splits: random; held-out *expressions*; **leave-one-lexeme-out** (LOO; the point stays in training through its other lexemes; evaluated only where possible); **held-out lattice points** (base points always in train). Controls: Hewitt–Liang control task under the same splits; matched α and dimensionality. Metrics: nearest-lattice-point accuracy, per-axis rounded accuracy, sign accuracy, R².
- **Geometry.** Centroid additivity ‖c(d1d2) − c(d1) − c(d2) + c(1)‖ vs random-triple baseline; dimensionless projection; named-vs-unnamed held-out points; transfer matrices (train condition → test condition); pre-answer-position transfer; Gemma Scope SAE reconnaissance.

## 2. Headline results

### 2.1 Decodability across lexemes and at the referent (H1) — P1b semantic set (12 lattice points, 68 lexemes), LOO-lexeme, chance ≈ 0.08
| model | unit L28: linear / categorical | anaphor L12: linear / categorical | control (linear, nearest) |
|---|---|---|---|
| Qwen3-4B (s0 / s1) | 0.51 / 0.55 ; 0.52 / 0.54 | 0.43 / 0.71 ; 0.41 / 0.68 | 0.04–0.07 |
| Qwen3-8B | 0.59 / 0.55 ; 0.49 / 0.55 | 0.42 / 0.71 ; 0.46 / 0.73 | 0.05–0.08 |
| Qwen3-14B | 0.49 / 0.50 ; 0.46 / 0.49 | 0.43 / 0.71 ; 0.45 / 0.72 | 0.05–0.07 |
| OLMo-3-7B | 0.55 / 0.54 ; 0.50 / 0.54 | 0.46 / 0.70 ; 0.46 / 0.72 | 0.05–0.07 |
| Gemma-2-9B | 0.43 / 0.44 ; 0.41 / 0.43 | 0.40 / 0.66 ; 0.41 / 0.66 | 0.04–0.07 |
Per-axis rounded accuracy 0.62–0.74 at all sites. Selectivity (linear task − control, nearest) at the anaphor: 0.33–0.40 in every model; categorical selectivity 0.58–0.67. For the three base dimensions alone (38 lexemes, 3-way), cross-lexeme accuracy is 0.83–0.93 at the unit token, 0.72–0.91 at mention end, 0.67–0.84 at the anaphor (Qwen3-4B; E1.1 diagnostic), and a base-dimension probe trained on mentions classifies the input-unit tokens of T1 prompts at 0.89–0.99 (E1.4). **Solid**: the dimension of a mentioned quantity is linearly decodable across unseen lexemes, is carried to the referent, and — for named derived dimensions — is best described categorically (one direction per dimension) rather than by exponent coordinates.

### 2.2 Lattice geometry (H2) — held-out named lattice points (P1b), five models × two seeds
Linear nearest-point accuracy on held-out points: **0.03–0.09** at every site (unit L16, anaphor L12; chance 0.08); per-axis sign accuracy 0.61–0.71; sign/magnitude and MLP parameterisations do not extrapolate either (E1.3). Centroid additivity residual 1.22–1.35 vs random-triple baseline 1.26–1.36 (no additive structure). Dimensionless mentions project near the origin in 0–1.5% of cases. In-distribution the linear-in-exponent probe reaches 0.82–0.92 exact at the unit token but 0.35–0.49 at the anaphor, where the categorical probe reaches 0.83–0.87. **Solid negative**: no linear exponent-lattice code that extrapolates; no vector-arithmetic identities; the origin is not special.

### 2.3 Composition at the pre-answer token (H4 precursor; Qwen3-4B, E1.4)
The answer dimension of T1v2 prompts is decodable at the pre-answer token within relation (random split nearest 0.78–0.80) but not across relations/lattice points (0.04–0.14, R² ≈ 0): a relation-specific code, not a composed lattice point.

### 2.4 Transfer (E1.2b, Qwen3-4B, P1-trained)
Compound-expression probes do not read named units (LATTICE→NAMED nearest 0.00–0.02): for compound strings the exponents are in the surface form, and that code is unrelated to the named-unit code — a stimulus lesson recorded the day it was found. REAL→cross-lingual nearest 0.14–0.20; REAL→invented 0.06–0.19 (best at the anaphor); neutral↔revealing nouns transfer well at the unit token (R² 0.7) and poorly at the anaphor (0.1–0.3).

### 2.5 Gemma Scope (E1.6, descriptive)
At the unit token every lattice point has a handful of near-exclusive latents (30–90% in-point, ≈0 elsewhere); 55 selected latents give LOO-lexeme nearest 0.54; selectivity is much weaker at the anaphor (0.28).

## 3. Anomalies and fixes (all logged the day found; none reinterpreted after the fact)
B1 origin-labelling of out-of-sublattice items (fixed before the headline sweep). B2 α fixed at 10 on 2,560 standardised features (replaced by CV). B3 compound-lattice stimuli measure surface-exponent readability, not dimension (retained as a control; semantic and P1b sets built). B4 5-fold lexeme folds degenerate into point holdout for named units (replaced by LOO-lexeme with the point retained). B5 layer index bug for 32-layer OLMo (negative indexing). B6 a swallowed assignment in the LOO script (rerun).

## 4. Pre-registered G1 rule and outcome
PASS required (≥2 models): cross-lexeme selectivity ≥ 0.30 at mention_end/anaphor **and** held-out lattice nearest ≥ 0.25 with R² ≥ 0.4. Observed: selectivity 0.33–0.40 (linear) / 0.58–0.67 (categorical) in all five models ✔; held-out lattice nearest 0.03–0.09 ✘ (R² negative). Distrust clauses for a FAIL do not apply (parameterisations tried; positions beyond `unit` show the effect). **Outcome: PARTIAL (H1 supported, H2 refuted for a linear lattice code).**

## 5. Calibration against PREDICTIONS_phase1.md (hit = within ±0.15 / right sign)
| # | predicted | observed (Qwen3-4B unless stated) | verdict |
|---|---|---|---|
| P1.1 unit cross-lexeme 0.70 (ctrl 0.10) | — | 0.48–0.55 (ctrl 0.05) | miss (over-predicted) |
| P1.2 mention_end 0.60 | — | 0.34 linear / 0.65 categorical | miss / hit |
| P1.3 anaphor 0.35 | — | 0.43 linear / 0.71 categorical | hit (exceeded) |
| P1.4 lattice holdout 0.35, R² 0.6 | — | 0.05, R² < 0 | miss |
| P1.5 named > unnamed extrapolation | yes | named worse (compound set) | miss |
| P1.6 additivity ratio 0.5 | — | ≈ 1.0 (= baseline) | miss |
| P1.7 dimensionless at origin ≥ 70% | — | 0–1.5% | miss |
| P1.8 REAL→INV-LEX 0.40 | — | 0.19 | miss |
| P1.9 REAL→XLING 0.55 | — | 0.14–0.20 | miss |
| P1.10 neutral↔revealing within 0.10 | yes | yes at unit, no at anaphor | partial |
| P1.11 noun-only decodes ≥ 0.5 | yes | 0.21–0.22 nearest (12-way) | miss |
| P1.12 categorical ≥ linear + 0.10; MLP ≤ linear + 0.10 | yes | categorical +0.12 to +0.50; MLP worse | hit |
| P1.13 pre-answer transfer ≥ 0.30 | — | 0.00–0.06 | miss |
| P1.14 larger models extrapolate better | +0.10 | no scaling (0.03–0.09 everywhere) | miss |
**What the misses imply.** I expected a partly linear lattice code and got a categorical dimension code; every prediction that assumed exponent coordinates (P1.4–P1.9, P1.13, P1.14) missed in the same direction. The framing changes accordingly: the object of Phase 2 is a categorical dimension variable (identity of the lattice point), and the "graded algebra" hypothesis (A2.5) is dropped in its geometric form — there is no gradient of a structure that does not exist; what may be graded is *categorical coverage* (which points get a direction), which Phase 2/3 can measure through per-point IIA and per-point decodability.
