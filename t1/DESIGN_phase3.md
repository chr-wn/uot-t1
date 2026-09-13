# DESIGN — Phase 3: generality, ecological validity, utility, writing (2026-09-13)

Scope is contained to what the ledger can carry; each item has a pre-registered expectation in PREDICTIONS_phase3.md.

- **E3.1 Naturalistic transfer.** Mine sentences with a numeral followed by a registry unit lexeme from real text (Wikipedia sample), label the dimension from the registry, cache Qwen3-4B residuals at the unit token and mention end, and apply the Phase-1 categorical/linear probes trained on synthetic mentions (P1b) without retraining. Report nearest-point / categorical accuracy vs a within-natural-text LOO-lexeme probe and the control task.
- **E3.2 Utility demo.** Probe-based dimension-inconsistency detector: for model-generated step-by-step solutions to unit-cloze problems (T1int, Qwen3-4B), read the dimension of the final stated unit with the Phase-1 categorical probe and compare with the expected dimension; corrupt half the solutions by replacing the final unit with a wrong-dimension unit. AUROC vs (i) an output-level self-check (ask the model "Is the unit consistent? yes/no", calibrated margin) and (ii) Pint parse-and-check (near-oracle for parseable units; undefined for invented units).
- **E3.3 Robustness (from existing runs).** Template spread (Phase 0), base vs instruct (Phase 0), tokenisation/surface style (Phase 0 style breakdown), seeds (Phases 1–2).
- **E3.4 Scaling.** Phase 1 already spans 4B–14B (+ OLMo-32B behavioural); no Phase-2 scaling beyond 4B/7B/9B (compute discipline).
- **Writing.** `reports/TECHNICAL_REPORT.md` (abstract, intro, related work, methods, results, limitations, threats to validity), `NEXT.md`, `Makefile` target that regenerates every figure from cached runs, environment capture.
