# DR-004 — G1 decision (2026-09-13)

**Pre-registered rule** (PREDICTIONS_phase1.md): PASS = cross-lexeme selectivity ≥ 0.30 at mention_end/anaphor (≥2 models) AND held-out lattice nearest ≥ 0.25 with R² ≥ 0.4; PARTIAL = first without second; FAIL = neither.

**Observed:** selectivity 0.33–0.40 (linear) and 0.58–0.67 (categorical) at the anaphor in all five base models, two seeds; held-out named lattice points 0.03–0.09 nearest with negative R² in every model and seed; additivity at baseline.

**Decision: PARTIAL → H1 supported (solid), H2 refuted for a linear exponent code (solid negative). Phase 2 proceeds as adjudication with the null (heuristics / dissociation) favoured over the algebra branch**, and the high-level variable for M_alg becomes *categorical* dimension identity with an equality mechanism (DESIGN_phase2 §1). Confidence in the decision: high.

**Declared deviations:** stimulus and probe fixes B1–B6 (gate report §3) all preceded the headline runs on the corresponding sets; thresholds untouched. The first DAS run (E2.1 pipeline test) happened before PREDICTIONS_phase2.md was written; its numbers are uninformative (alg training at chance; heur "IIA" equal to the majority rate under an unbalanced label set) and are logged as such; PREDICTIONS_phase2.md is committed before the corrected sweep is read.

**What would reverse this:** a named-unit set with ≥ 30 lattice points showing linear extrapolation ≥ 0.25; or a parameterisation (e.g. per-axis one-hot magnitudes) that extrapolates where the tried ones failed.

**Rejected branch steelman (FAIL):** one could argue the referent-position decodability is a copy of the lexeme's representation. Rejected because the referent code differs qualitatively from the unit-token code (linear collapses, categorical persists; noun-family transfer changes), and because the pre-registered rule keys on selectivity, which is met by a wide margin.
