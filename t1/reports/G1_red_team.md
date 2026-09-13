# G1 red-team appendix (2026-09-13)

Reading under attack: *"dimension is represented as a categorical variable (one direction per lattice point), shared across lexemes and carried to the referent; there is no linear exponent-lattice code."*

## Attack 1 — "No linear lattice code" is a failure of my parameterisation or probe, not of the model
Tried: linear-in-exponent, sign/magnitude, MLP, categorical; PCA-512 with CV α; two seeds; five models; two stimulus sets. Held-out points fail under every parameterisation while in-distribution linear decoding at the unit token is 0.82–0.92 — the failure is specific to extrapolation, not to fit. **Residual risk:** the held-out-point test has only 12 points and 5 folds; with so few points a true but noisy linear code could fail to extrapolate. Mitigation available: the compound-lattice set (124 points) also failed to extrapolate to *named* points (R² −2) while extrapolating to unnamed compound points (R² +0.5) — the opposite of what a semantic linear code predicts. Not fully excluded; a larger named-point set would need units for exotic dimensions (there are few).

## Attack 2 — The categorical code is lexical after all
Cross-lexeme LOO holds out one lexeme while other lexemes of the same point remain; a probe could exploit *shared substrings* (kilojoule/megajoule; kilopascal/megapascal). **Partly excluded**: the base-dimension 3-way test uses lexically unrelated lexemes (meter/mile/inch/fathom…) at 0.67–0.93; per-point LOO accuracies are also high for points with heterogeneous lexemes (pressure: pascal/psi/bar/torr/atmosphere → categorical 0.90 at the anaphor). **Not excluded** for derived points with prefix families; a lexeme-family-out fold (hold out all SI-prefixed variants together) is a cheap follow-up (Phase 3 robustness).

## Attack 3 — The referent-position signal is a copy of the unit token, not an attached property
If attention simply copies the unit token's representation to the anaphor, "attached to the referent" is over-interpreted. Evidence for something more than copying: at the referent the *linear* exponent readout collapses (0.35–0.49) while the categorical one stays high (0.83–0.87), and neutral→revealing transfer drops from R² 0.7 (unit) to 0.1–0.3 (anaphor) — the referent representation is re-coded, not copied. **Partly excluded**; a causal test (patching the referent vs the unit token) belongs to Phase 2.

## Attack 4 — Decodable ≠ used
Nothing in Phase 1 shows the categorical variable is read by the model's judgments. **Conceded**; this is exactly Phase 2's question, and the pipeline-test DAS run (§Notebook, E2.1 first pass) already warns that a rank-16 M_alg subspace at layer 16 of the first unit token did not train above chance — the causal branch may well be "dissociation".

## Attack 5 — Stimulus artefacts
The compound-lattice set turned out to measure surface-form exponent reading (found by the transfer test, logged, and demoted to a control). The semantic sets could carry a subtler artefact: named-unit *frequency* (rare units like poundal, sverdrup) could fail for lack of knowledge, not lack of representation. Per-point LOO shows the hardest points are area/volume/dose (few natural lexemes) rather than the rare-lexeme ones; recorded as a limitation.

## Verdict
The PARTIAL decision is sound. The strongest remaining threat is Attack 1 (small named-point set for extrapolation). The cheapest discriminator still unrun: a lexeme-family-out fold (Attack 2) and a referent-vs-unit patching test (Attack 3), both scheduled.
