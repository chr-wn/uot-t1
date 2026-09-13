# NEXT — what I would do with more time (2026-09-13)

## Weakest ledger claim and how to shore it up
**C13/C15 (the licensing mechanism is a pairwise heuristic; invented units untouched at the lexeme token).** The invented-unit null is site-limited: their dimension lives in the definition sentence and at the referent. Run DAS at (a) the definition's kind-word token ("unit of *length*"), (b) the referent/anaphor token, (c) quantity 2's unit token with quantity 1 as source. If a low-rank subspace at (a) or (b) flips judgments for invented units per dimension equality (IIA ≥ 0.65) with random ≤ 0.55, the categorical dimension variable *is* causally used and C13 must be weakened to "at the lexeme token". Cost: 2 GPU-hours.

## Ordered list
1. Definition/referent-site DAS for invented units (above).
2. Task calibration: repeat Phase 2 with the `convert` operation only, and with 3 in-context examples, to raise the equality ceiling (currently 0.65–0.67 bounds M_alg); also instruct models with a generation readout.
3. Lexeme-family-out folds (hold out all SI-prefixed variants together) for the LOO-lexeme decodability claim (C7).
4. A larger named-point set for lattice extrapolation (C8 has 12 points): add named units for L²/T, M/(L·T), L·T, I·T-type points via context definitions.
5. Composition with a lawful counterfactual candidate in the set (H4), and mid-CoT interventions on reasoning models.
6. Value–dimension double dissociation with a purer intervention (rank-16 subspace orthogonalised against the value probe).
7. Selectivity matrix as pre-registered (6×6 implanted × base dimension per operation) rather than per-dimension IIA.
8. Utility: extend the inconsistency detector to naturally occurring errors in long CoTs (not synthetic corruption).

## Three attacks a skeptical reviewer will make, and my current best answers
1. *"Your equality model was handicapped by a task the models do badly; the null is about behaviour, not representation."* Partly true: IIA under equality labels is bounded by the 0.65 ceiling and the subspace reaches 92% of it. But the *comparison* is fair at matched rank, and the discriminating tests (disagree subset, label swap, invented units, value contamination) all point the same way; item 2 above is the direct fix.
2. *"Decodable across lexemes just means lexically similar lexemes share embeddings."* Base dimensions decode across lexically unrelated lexemes (meter/mile/inch/fathom) at 0.83–0.93; per-point LOO is high for heterogeneous points (pressure: pascal/psi/bar/torr/atmosphere → 0.90). Item 3 closes the remaining gap.
3. *"Twelve points is too few to call the lattice test negative."* Conceded as a limitation; the compound-lattice set (124 points) fails to extrapolate to *named* points while extrapolating to unnamed compounds, which is the wrong direction for a semantic linear code, and additivity is at baseline on both sets. Item 4 is the cheap fix.
