# DR-001 — Adversarial pass over the Phase-0 design (2026-09-12)

## Worst flaw found
**The compositional unit-of-answer cloze can be solved by surface copying.** In "travels 12 blorks in 3 zims, so its speed is 4 ___", the answer "blorks per zim" is recoverable by the template rule *first-unit per second-unit* without any representation of dimension. A model that has memorised the pattern "travels A in B → A per B" would pass the G0 criterion as originally shaped ("well above chance on compositional unit-of-answer with novel combinations"), and Phase 1 would then be built on a result that is A1 (template retrieval) rather than composition.

## Fix adopted
1. Half of all T1 items are **order-swapped**: the inputs appear in the surface order opposite to the composition order ("It took 3 zims to cover 12 blorks; the speed is 4 ___"). Copying then yields "zims per blork", which is a candidate distractor.
2. The candidate set always contains the **template-copy distractor** explicitly, so the readout separates copying from composing.
3. Relations with non-trivial composition (density from mass and a *side length* → M/L³; pressure from force and area given as a side length → M/(L T²)) are included so that composition requires more than concatenating the input lexemes.
4. G0 requires the order-swapped subset to be above chance on its own (see `PREDICTIONS_phase0.md`).

## Second flaw (fixed)
Probe positions. Decoding dimension at the unit token is a non-result (brief requirement 1), but the Phase-1 position list omitted the *pre-answer* token of the cloze — the one place where a *composed* dimension must be represented if the model composes. Added as a position class.

## Third flaw (mitigated, not removed)
Frequency and lattice distance are confounded for familiar units (unnamed lattice points are also low-frequency). Mitigation: INV-LEX items have zero corpus frequency for the composed string *by construction*, so within INV-LEX the distance effect is measured without frequency confounding; within FAM, both predictors enter one regression and we report partial effects, and we include FAM-UNNAMED points whose component unit strings are frequent.

## Residual risks accepted
- Invented lexemes are defined in context; failures there can be binding failures, not algebra failures. Twins (requirement 6) separate these only partially; Phase 2's binding-contamination checks are the real test.
- Base-model log-prob readouts over multi-token candidates favour short candidates; mean-per-token log-prob (as the rival paper) reduces but does not eliminate length bias. We will report a length-matched subset.

## Addendum (2026-09-13, after inspecting generated items, before any model run)
- **Length confound found and fixed.** With the original distractor set (inverted, exponent-dropped, product, single unit) the correct candidate was the longest string in 62% of T1 items and never the shortest. Distractors are now built from *all* lexemes of the correct answer with different exponents (inverted, exponent-permuted, random lattice point), so candidates differ only in exponents/arrangement. The `single`/`product` roles remain only as last-resort fallbacks.
- **Unnatural unit pairings** ("126 yd in 9 ms", "foot kilograms") replaced by natural, system-consistent choices for T1 and for all compound compositions. T2/T4/T5 single-unit picks still draw from the broader core set (ms, yr, oz …) because plausibility matters less for yes/no judgments and the broader set widens lexeme coverage.
- Neither change touches a threshold in PREDICTIONS_phase0.md.
- **Final length statistics (T1_s0, 1440 items, after the exponent-magnitude-matched sampler):** correct candidate strictly shortest in 14%, strictly longest in 1%, tied/middle 75%, all four equal 10%; mean length 13.2 chars (correct) vs 14.3 (distractors). Residual bias is small but nonzero; as pre-registered, the gate report also reports the length-matched subset (items where the correct candidate is not strictly shortest/longest, n≈1223).
