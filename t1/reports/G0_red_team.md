# G0 red-team appendix — the strongest case that the headline reading is wrong (2026-09-13)

Headline reading under attack: *"all six base models compose the unit of a derived quantity for novel lexemes and unnamed lattice points (≈0.9), so a compositional dimension variable is behaviourally licensed."*

## Attack 1 — The task is verbal, not compositional
Every T1 prompt names or states the relation ("speed", "mass squared times length"), and the invented-unit prompts define each lexeme's dimension in words. E0.3b shows that with the scenario's *units deleted*, models still reach 0.38–0.86. The candidate set is small (4) and structured; picking "X per Y" for "speed" requires only that the model map relation words to a template over the lexemes that appear in the context. Nothing in T1 forces a dimension to be attached to the *mentioned quantities*. **Not excluded.** Cheapest discriminating experiment: (i) a "definition-by-example" cell where the relation is never named or verbalised (Phase-1 E1.0b: "Q = 12 blorks × 3 zims = 36 ___" style, and multi-step scenarios where the derived quantity is computed from intermediate quantities); (ii) representational tests at the quantity tokens (Phase 1). The full-prompt increment over the no-units control (+0.08 to +0.47) is evidence that *something* about the mentioned quantities matters, but it could be lexeme availability/recency rather than dimension.

## Attack 2 — Candidate readout inflates competence
Free generation is much worse: only 42–83% of generations that contain a unit have the right dimension, and 4–33% contain no unit at all. Errors are dominated by dropped exponents ("snoorp per shoath" for a force). Under free generation the honest number for INV-LEX composition is ≈0.5–0.7, not 0.9. **Partly conceded.** The pre-registered primary readout is the candidate one, and the gate passes under both (generation-based accuracy on INV-LEX is still ≥ 0.4 above chance), but the *paper's* behavioural claim must report both and must not say "≈0.9 accuracy" without the qualifier. The exponent-dropping pattern is itself a finding for Phase 1 (is exponent magnitude represented?).

## Attack 3 — Distractor structure leaks the answer
Candidates share all lexemes and differ only in exponents/signs; the correct answer is the one whose exponent pattern matches the relation words. A model that recognises "per" ↔ "divided by" and "squared" ↔ "squared" solves the task lexically. **Partly excluded**: order-swapped items (where surface order contradicts composition order) are not harder, and the "inverted" distractor has the same words in the other order; but the "exp_swapped" distractor is where most residual errors go (Q(M²L): "grams per square…"), consistent with a partly lexical strategy. Discriminator: Phase-1 probes at the pre-answer token must decode the composed dimension *before* any candidate is seen.

## Attack 4 — Familiar twins are not matched for prior
Twin items use real units, so the candidate prior is not neutral (no-context accuracy 0.02–0.21 for twins, i.e. the prior *disfavours* the correct twin string) whereas for invented items it is 0.41–0.46. The equal accuracies of invented and twin cells therefore compare different prior baselines. **Conceded; consequence small**: it means invented-unit performance is *not* inflated relative to twins by the prior — if anything twins are handicapped — so the direction of any bias runs against the headline, not for it.

## Attack 5 — Length/tokenisation residuals
v2 candidates are exponent-magnitude-matched and prefix-free, but the correct string is still strictly shortest in 12% of items. Length-matched subset accuracies equal full-cell accuracies within 0.01. **Excluded** for T1.

## Attack 6 — Two-way and three-way judgment results are readout artefacts
Constant yes/no and letter biases were found (A3, A5). AUROC/calibrated readouts show real ranking signal for T2/T4 (invented > familiar near-miss), but the raw numbers cannot be quoted as accuracies. **Conceded**; the paper reports AUROC/calibrated accuracy for these tasks and uses instruct models with a generation readout for any accuracy claim.

## Attack 7 — The "no gradient with distance/frequency" is a ceiling effect
Accuracy is 0.85–0.98 everywhere; a gradient cannot show at the ceiling of a 4-way task. **Conceded**: absence of a behavioural gradient is not evidence against A2.5; it shows only that this battery cannot see it. The gradient must be sought where errors exist (free generation; held-out-lattice probes).

## Verdict of the red team
The PASS is sound on the pre-registered criterion, but the headline must be worded as *"models select the dimensionally correct composed unit for novel lexemes and unnamed lattice points far above chance, and a large part of that selection is achievable from relation wording alone"*. The decisive question — whether the mentioned quantities carry a dimension variable that is composed — is untouched by Phase 0 and is exactly what Phase 1 must test at the quantity/anaphor/pre-answer positions with cross-lexeme and lattice-holdout probes. Cheapest addition to run early in Phase 1: the definition-by-example cell (Attack 1) and a *free-generation* twin of T1 with the number removed from the prompt so the model must emit the unit (fixes A4).
