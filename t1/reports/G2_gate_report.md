# G2 gate report — Phase 2 causal adjudication (2026-09-13)

Companions: `reports/G2_red_team.md`, `decisions/DR-005-G2-decision.md`, `PREDICTIONS_phase2.md`, `figures/phase2/`, `runs/E2.*`.

## 1. What ran
- **Task and stimuli.** Two-quantity yes/no consistency judgments (add / compare / equate / convert; 2 templates each), single-token unit surface forms (43 lexemes over 10 lattice points, audited on the tokeniser), values 2–99. Interchange examples: 4,800 real (base/source pairs, balanced so that half the sources share quantity 2's dimension), 1,600 invented-unit twins (dimension defined in context), train/eval split by base item; flagged subsets: *disagree* (M_alg and M_heur labels differ), *same-dimension lexeme swap*, *held-out lexemes* (never in DAS training).
- **Rival high-level models.** M_alg: D1 (categorical dimension of quantity 1), answer = yes iff D1 = D2. M_heur: U1 (unit lexeme), answer = the model's own pairwise table (52×52 lexemes × 4 ops, 16k prompts; AUROC vs dimension equality 0.89–0.98; rank-calibrated threshold at the true same-dimension rate). Counterfactual labels are computed by each model, never by hand.
- **Interventions** (own implementation, unit-tested): residual-stream interchange at the unit-1 token (u1), unit-2 token (u2) and answer token, layers 4–28; learned orthogonal subspaces of rank 1–128 (Adam on an orthogonal parametrisation, 1,000 steps, lr 5e-3), matched-rank random subspaces, probe-derived subspaces (Phase-1 logistic-16 and ridge-3 directions), and the full-residual patching ceiling. **Balanced IIA** = mean of IIA on output-changing and output-preserving examples relative to the model's *own* base prediction (chance 0.50).
- **Controls:** 4 seeds at the headline site; natural-distribution projection (Makelov); value-preservation probe; binding (D2 readout after a D1 swap); noising; dose–response; text-rewrite baseline; per-point and per-operation breakdowns.

## 2. Where quantity 1's unit is causally read (ceiling)
Replacing the *entire* residual of the u1 token: layer 4 → balanced IIA 0.67 under M_alg labels (40% lawful flips) and 0.85 under M_heur labels (73%); layer 8 → 0.65 / 0.78; layer 12 → 0.57 / 0.63; layer ≥ 16 → 0.50 (no effect). u2: 0.52–0.55 / 0.63–0.66 at layers 4–16. Answer token: nothing until layers 24–28, where a late decision variable flips 28–48% (this is the answer, not D1). Invented units: 0.50 at every site — their dimension is carried by the definition sentence, not the lexeme token. **The model's judgments follow its pairwise table far more than dimension equality even under full patching** (0.73 vs 0.40 flips): the equality model is a worse description of the behaviour than the lexical table, before any subspace question arises.

## 3. Headline: matched-rank subspaces at u1 (Qwen3-4B-Base; Fig. P2a)
| rank | M_alg learned (L4 / L8) | M_heur learned (L4 / L8) | random | probe-derived (L8): alg / heur |
|---|---|---|---|---|
| 1 | 0.50 / 0.50 | 0.51 / 0.52 | 0.50 | — |
| 3 (ridge exponent basis) | — | — | — | 0.50 / 0.51 |
| 4 | 0.51 / 0.51 | 0.53 / 0.53 | 0.50 | — |
| 16 | 0.53 / 0.57 | 0.58 / 0.60 | 0.50 | 0.54 / 0.57 |
| 64 | 0.58 / 0.60 | 0.61 / 0.64 | 0.50 | — |
| 128 | 0.63 / 0.61 | 0.70 / 0.65 | 0.50 | — |
| full (2560) | 0.67 / 0.65 | 0.85 / 0.78 | — | — |
Seed means ± sd (4 seeds, rank 64, L8): M_alg 0.595 ± 0.003, M_heur 0.621 ± 0.027, random 0.499 ± 0.003; disagree 0.44 ± 0.04 vs 0.52 ± 0.02; held-out lexemes 0.61 / 0.63 (no memorisation of lexemes); invented 0.50 / 0.50; same-dimension lexeme swaps 0.51 / 0.53; the alg-trained subspace scored with heur labels 0.66 (> its own 0.60). Training accuracy of the alg subspace never exceeds 0.62 (heur 0.78). Per implanted dimension: base 0.57–0.58, derived 0.60 (no gradient); per operation: convert 0.70, compare 0.60, equate 0.55, add 0.52.

## 4. Controls (u1 L8, rank 64, 3 seeds; downstream probes at block 14)
| control | M_alg subspace | M_heur subspace | random rank-64 |
|---|---|---|---|
| natural projection: LOO-lexeme D1 accuracy on the projection (majority 0.12) | 0.69 | 0.69 | 0.47 (random 64-dim projection 0.55) |
| value probe R² (quantity 1's numeral): clean → after swap | 0.96 → 0.55 | 0.96 → 0.74 | 0.96 → 0.93 |
| D2 readout at u2: clean → after swap | 1.00 → 0.99 | 1.00 → 1.00 | 1.00 → 1.00 |
| noising: clean "yes" + mismatching source → "no" | 0.49 | 0.36 | 0.02 |
| dose–response α = .25/.5/.75/1 | .50/.53/.57/.60 | .51/.53/.55/.62 | .50 |
| lawful changes, subspace vs text rewrite (same examples) | 0.25 vs 0.42 | 0.17 vs 0.42 | 0.02 vs 0.42 |
The subspace is real (not dormant), necessary (noising), dose-monotone, and does not contaminate quantity 2, but it is **not value-free** (P2.11 fails): the swap disturbs the downstream numeral readout of quantity 1.

## 5. Replication
- **OLMo-3-7B** (u1 L4; one seed): ceiling 0.62 (alg) / 0.72 (heur); rank-64 learned 0.57 / 0.58; random 0.50; held-out lexemes 0.58 / 0.64; invented 0.50; disagree 0.44 (alg) vs 0.56 (heur); alg subspace under heur labels 0.62. Same pattern; effects concentrate at layer 4 (32-layer model).
- **Gemma-2-9B:** base accuracy on these prompts 0.51 (constant "yes" bias, cf. Phase 0); every intervention including the full-residual ceiling is at 0.49–0.51 under the raw readout — the task cannot be adjudicated for this model with an argmax readout. **Bias-corrected readout** (yes iff the yes–no logit margin exceeds the clean median; base accuracy becomes 0.60): ceilings at u1 L4 0.60 (alg labels) / 0.69 (heur); learned rank-64 subspaces at L4: **M_alg 0.52 (random 0.49) vs M_heur 0.67** (≈ its ceiling); disagree subset 0.44 vs **0.68**; held-out lexemes 0.53 / 0.64; invented 0.50 / 0.49; L8: 0.52 / 0.63. Gemma-2-9B is the clearest case: the equality subspace is indistinguishable from random while the pairwise-table subspace reaches its ceiling and generalises to unseen lexemes.

## 6. Composition (H4, exploratory, E2.5)
Implanting slot 2's dimension into slot 1 of a composition prompt (full residual, u1, L4/L8): the correct composed candidate loses 1.7–1.8 nats (text rewrite −1.9), argmax flips 7–14% (rewrite 18%), matched-rank random 0.0 nats — the composed answer depends causally on slot 1's unit representation, but no lawfully composed alternative was in the candidate set, so lawful propagation is not scored (n = 44).

## 7. Pre-registered decision rules — evaluation [FINAL]
- Algebra/categorical-causal: requires M_alg ≥ 0.70 with invented ≥ 0.65 and held-out ≥ 0.65 and alg ≥ heur on disagree — **not met** (best alg 0.63; invented 0.50; disagree alg < heur).
- Heuristics-extension: M_heur ≥ M_alg + 0.10 on disagree with M_heur ≥ 0.65 and alg invented ≤ 0.55 — Qwen3-4B: gap 0.06–0.14, M_heur 0.54–0.63 (borderline); OLMo-7B: gap 0.12, M_heur 0.56 (borderline); **Gemma-2-9B (bias-corrected): gap 0.24, M_heur 0.68, invented 0.50 — met in full.**
- Dissociation: ceiling ≥ 0.75 but neither subspace > random + 0.10 at rank ≤ 64 — **not met** (heur 0.64 = random + 0.14 at rank 64).
- Graded: alg IIA ≥ 0.70 for base but ≤ 0.55 for derived — **not met** (0.57–0.58 vs 0.60).
The heuristics-extension criterion is met in full for one model (Gemma-2-9B, bias-corrected readout) and directionally for the other two; no other branch is met for any model (DR-005 states the decision, the shortfalls, and the algebra steelman).

## 8. Calibration against PREDICTIONS_phase2.md (hit = within ±0.10 / right sign)
| # | predicted | observed | verdict |
|---|---|---|---|
| P2.1 ceiling at u1 early layers 0.85 | — | 0.67 (alg labels) / 0.85 (heur labels) | hit only under heur labels |
| P2.2 M_alg rank 64 0.62 | — | 0.60 | hit |
| P2.3 M_heur rank 64 0.66 | — | 0.62 | hit |
| P2.4 random 0.50 ± 0.03 | — | 0.50 | hit |
| P2.5 M_alg invented 0.55 | — | 0.50 | hit (null) |
| P2.6 held-out within 0.05 of all | — | +0.01 / +0.01 | hit |
| P2.7 heur > alg on disagree by 0.10 | — | +0.08 (rank 64), +0.14 (rank 128) | hit |
| P2.8 probe-derived ≥ 80% of learned rank-16 | yes | ≈ 90% | hit |
| P2.9 dose–response monotone | yes | yes | hit |
| P2.10 selectivity-matrix contrast 0.25 | — | not run as a matrix; per-dimension IIA flat (0.57–0.67) | not scored |
| P2.11 value preserved within 0.05 | — | R² 0.96 → 0.55 | miss |
| P2.12 lawful composition shift | no | no lawful target available; old answer suppressed as with text edit | hit (null) |
Nine hits, one miss, one unscored, one conditional. The miss (value preservation) is the informative one: it says the causally effective object at the unit token is a bound number–unit representation, which is also why the two rival models are hard to separate at this site.
