# DESIGN — Phase 2: causal adjudication (draft written 2026-09-13 before the G1 decision record; finalised with it)

## 0. What Phase 1 hands over (provisional)
Across five base models: the dimension of a mentioned quantity is decodable across lexemes (strongly for base dimensions, moderately for named derived dimensions), it is carried to the referent (anaphor) position, and it is represented *categorically* (one direction per lattice point) rather than as a linear exponent code: held-out lattice points do not extrapolate, centroids are not additive, and the composed answer dimension at the pre-answer token is relation-specific. So the causal question is not "is the lattice code used" but **"is a categorical dimension variable — as opposed to unit-lexeme identity — what licenses operations?"** H2 in its linear form is not the foundation; H3 is tested with a categorical high-level variable.

## 1. Rival causal models (both fully specified, both aligned by DAS at matched rank)
Task family for alignment: two-quantity consistency judgments (T2-style, single-token yes/no answer), base models, calibrated by median margin.
- **M_alg (categorical dimension):** variables D1, D2 ∈ {lattice points}; mechanism: answer = yes iff D1 = D2. Counterfactual label after interchanging D1 from a source: 1[D1(src) = D2(base)].
- **M_heur (lexical / pairwise cue):** variables U1, U2 ∈ {unit lexemes}; mechanism: answer = compat(U1, U2), where compat is the model's *own* behavioural table (estimated from E0.3 T2/T4 margins, thresholded at the calibrated median). Counterfactual label after interchanging U1: compat(U1(src), U2(base)).
The two models agree on most interchanges; the discriminating set is where they disagree: (a) source unit is a different lexeme of the *same* dimension as the base's U1 (M_alg: label unchanged; M_heur: label follows the pair table, which is often wrong for near-miss/rare pairs); (b) invented lexemes defined in context (M_heur has no entry; M_alg predicts by the defined dimension); (c) held-out lexemes never seen during DAS training (a subspace that generalises across lexemes is dimension-like).

## 2. Interventions (own implementation, `uot.das`)
Site: residual stream at the last token of quantity 1's unit and at the mention-end token (Phase-1 best sites), layers swept over Phase-1 peaks (±4). Learned orthogonal subspace of rank k ∈ {1, 2, 4, 8, 16, 32, 64, 128} for each rival variable; IIA on held-out interchange examples with balanced output-changing/preserving pairs (chance 0.5).
Controls (all mandatory): matched-rank random subspaces (10 seeds); probe-derived subspaces (Phase-1 ridge/logistic directions, orthonormalised) at the same rank; rank sweeps; denoising (implant D from a source into a corrupted base) and noising (implant a mismatching D into a clean base); dose–response (alpha ∈ {0, .25, .5, .75, 1}); value-preservation (a numeral-value probe trained in Phase 1 must read the same value after the dimension swap); binding contamination (swapping D1 must not change a D2 readout; position-swap controls per Feng & Steinhardt); natural-distribution projection check (Makelov): the found subspace must separate the natural conditions when natural activations are projected on it; full-layer patching as the ceiling; text-rewrite baseline (edit the unit string in the prompt) reported next to every intervention effect.

## 3. Selectivity matrix (pre-registered in PREDICTIONS_phase2.md before any headline run)
Operations: add, compare, equate, convert; implanted dimensions: L, M, T, L/T, force, energy (with base dimension of quantity 2 fixed per cell). Algebra-predicted legality: yes iff implanted D1 equals D2. Prediction per cell: yes-rate after intervention.

## 4. Composition (H4, reduced scope after E1.4)
Since no composed lattice point was found at the pre-answer token, H4 is tested at the *output* level only: intervene on input quantity 1's dimension subspace in a T1 prompt and check whether the candidate ranking moves toward the lawfully composed unit (e.g. implanting "time" into the distance slot of a speed problem should favour "s/s"-type or dimensionless answers). Pre-registered as exploratory; a null here is consistent with Phase 1.

## 5. Experiment list
| ID | What |
|---|---|
| E2.0 | interchange stimulus builder (base/source pairs, both counterfactual label sets, discriminating subsets, invented-unit twins) |
| E2.1 | DAS M_alg vs M_heur, Qwen3-4B, rank sweep, IIA on all / discriminating / held-out-lexeme / invented subsets |
| E2.2 | control regime (random, probe-derived, denoise/noise, dose, value preservation, binding, natural-projection, full-layer, text-rewrite) |
| E2.3 | selectivity matrix |
| E2.4 | replication on OLMo-7B, Gemma-9B (Qwen3-14B if time) |
| E2.5 | H4 output-level composition test |
| G2 | gate report, red team, decision record: algebra / dissociation / heuristics-extension / graded |
