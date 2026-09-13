# DR-003 — G0 decision (2026-09-13)

**Pre-registered criterion** (PREDICTIONS_phase0.md, committed before any run): PASS if for at least one base model (a) T1 INV-LEX acc_cand ≥ 0.50 with CI lower ≥ 0.40, (b) order-swapped INV-LEX ≥ 0.40 with CI lower ≥ 0.30, (c) FAM-UNNAMED ≥ 0.50.

**Observed** (T1 v2, n=480 per cell): (a) 0.89–0.97 (CI lower ≥ 0.84), (b) 0.89–0.95, (c) 0.84–0.91, for all six base models. Distrust clauses (swap penalty > 0.30; length-matched subset below threshold; generation at chance) not triggered.

**Decision: PASS → proceed to Phase 1 on the full base suite.** Confidence that the criterion is met: high (solid). Confidence that the *interpretation* "compositional dimension variable exists" is supported by Phase 0: low — deliberately; Phase 0 was a competence gate, and the red team shows a verbal route (relation words + candidate structure) explains a large share of the competence.

**Deviations from pre-registration (declared, not reinterpretations):** stimulus-validity fixes before/after the first runs (A1 lexeme leak; A2 prefix candidates → v2), all pre-allowed by the "distrust a FAIL" clause and none touching thresholds; added readouts (dimension-parsed generation, AUROC/calibrated accuracies) alongside — never replacing — the pre-registered primary readout. v1 results are retained and reported.

**What this does to the plan.**
1. Phase 1 must locate the dimension of the *mentioned quantities* (unit / mention-end / anaphor positions) and of the *composed* answer (pre-answer position), with cross-lexeme and lattice-holdout generalisation — the behavioural battery cannot separate A1 (verbal template) from A3 (algebra).
2. Add two cheap behavioural cells early in Phase 1 (E1.0b): definition-by-example prompts (relation never verbalised) and a free-generation twin of T1 with the output number removed.
3. The graded-boundary hypothesis (A2.5) is not testable behaviourally with 4-way candidates at this range (ceiling); it moves to representations (which held-out lattice points extrapolate) and to free-generation exponent errors.
4. Judgment tasks: report AUROC/calibrated accuracy; use instruct models with a generation readout for accuracy claims.

**What would reverse this decision:** discovering that the full-prompt increment over the no-units control is explained by lexeme recency/availability rather than dimension (a Phase-1 control: scenario with units *swapped between slots* — if accuracy is unchanged, the units' binding to quantities is not used); or free-generation accuracy at chance in a no-candidate setting.

**Rejected branch steelman:** "FAIL/PARTIAL because the competence is verbal." I reject this because the criterion was pre-registered as a competence criterion and is met by a wide margin; reinterpreting it post hoc would be exactly the metric-shopping the brief forbids. The concern is preserved as the central open question rather than as a gate failure.
