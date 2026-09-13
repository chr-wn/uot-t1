# PREDICTIONS — Phase 2 (committed 2026-09-13 before the corrected DAS sweep is read; never edited after)

Declared: one pipeline-test DAS run (Qwen3-4B, L16, u1, rank 16, 300 steps, lr 1e-3) preceded this file; it was uninformative (M_alg train acc 0.48; M_heur "IIA" 0.74 = majority rate under unbalanced labels; random 0.50) and led to two fixes (balanced base-prediction-relative IIA; rank-calibrated M_heur threshold). No other Phase-2 result has been seen.

## Setup
Two-quantity yes/no judgments (add/compare/equate/convert), Qwen3-4B-Base first; variable = quantity-1's dimension (M_alg, categorical; mechanism: yes iff D1 = D2) vs quantity-1's unit lexeme (M_heur; mechanism: the model's own rank-calibrated pair table). DAS (own implementation) at the residual stream of the u1 token / u2 token / answer token, layers {4…28}, ranks {16, 64} (sweep) then {1,2,4,8,…,128} at the best site. Balanced IIA = mean over output-changing and output-preserving examples relative to the model's own base prediction (chance 0.5). Controls: matched-rank random subspaces, probe-derived subspaces, full-residual patching ceiling, dose–response, noising, value preservation, binding (D2 readout unchanged), natural-distribution projection, text-rewrite baseline.

## Point predictions (Qwen3-4B-Base, best site)
| # | prediction | value |
|---|---|---|
| P2.1 | full-residual patching ceiling at u1 (early layers 4–12) | balanced IIA 0.85 |
| P2.2 | M_alg learned subspace, rank 64, best site | 0.62 |
| P2.3 | M_heur learned subspace, rank 64, best site | 0.66 |
| P2.4 | matched-rank random | 0.50 ± 0.03 |
| P2.5 | M_alg on invented-unit examples (definitions in context) | 0.55 |
| P2.6 | M_alg on held-out lexemes | within 0.05 of all-example IIA |
| P2.7 | on the disagree subset, M_heur > M_alg | by 0.10 |
| P2.8 | probe-derived (logistic-16) subspace at rank 16 reaches ≥ 80% of the learned M_alg IIA | yes |
| P2.9 | dose–response monotone in α for whichever subspace works | yes |
| P2.10 | selectivity matrix (implanted D1 × base D2, 6×6, per op): diagonal yes-rate − off-diagonal yes-rate | 0.25 |
| P2.11 | value-preservation: numeral-value probe accuracy after dimension swap | within 0.05 of no-swap |
| P2.12 | H4 output-level composition (implant "time" into a distance slot of a speed problem): lawful shift of candidate ranking | no (null) |

## G2 decision rules (pre-registered)
- **Algebra/categorical-causal branch** if, for ≥ 2 models: M_alg balanced IIA ≥ 0.70 at some site with matched-rank random ≤ 0.55, **and** M_alg IIA on invented units ≥ 0.65 **and** on held-out lexemes ≥ 0.65, **and** M_alg ≥ M_heur on the disagree subset. (Linear-lattice claims are not on the table after G1; "algebra" here means a dimension-identity variable with an equality mechanism.)
- **Heuristics-extension branch** if M_heur ≥ M_alg + 0.10 on the disagree subset with M_heur ≥ 0.65 and M_alg on invented units ≤ 0.55.
- **Dissociation branch** if the full-residual ceiling ≥ 0.75 at some site but neither learned subspace exceeds random + 0.10 at rank ≤ 64: the variable is decodable (G1) but not read out through a low-rank subspace at any single site.
- **Graded branch** if IIA under M_alg is ≥ 0.70 for base dimensions but ≤ 0.55 for derived dimensions (per-point IIA), i.e. causal use exists only near the origin.
- Any branch is reported with the rejected branch steelmanned.

## What would make me distrust an algebra-branch result
Makelov dormancy: a learned subspace whose natural-distribution projection does not separate D1 classes; or IIA driven only by output-preserving examples (report changing/preserving separately); or IIA that disappears under noising.
## What would make me distrust a dissociation result
Training failure: report train accuracy, try lr {1e-3, 5e-3, 2e-2}, steps 1000–3000, ranks up to 256, and the probe-derived subspace; if all remain at chance while the ceiling is high, dissociation stands.

## Calibration commitment: score P2.1–P2.12 at G2.
