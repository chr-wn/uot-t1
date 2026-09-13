# PREDICTIONS — Phase 1 (committed 2026-09-13, before any Phase-1 cache/probe run; never edited after)

Setting: residual-stream probes (ridge → exponent vector over L, M, T) on single-quantity mentions (`data/phase1/P1_s{0,1}.jsonl`), Hewitt–Liang control task, cross-lexeme and lattice-holdout splits; positions value / unit / mention_end / anaphor / last; layers every 2. Nearest-lattice-point accuracy ("nearest") among 124 REAL-LATTICE points unless stated; chance ≈ 0.01–0.03.

## Point predictions (Qwen3-4B-Base; best layer)
| # | prediction | value |
|---|---|---|
| P1.1 | cross-lexeme nearest at `unit` position (REAL-LATTICE, neutral family) | 0.70; control 0.10 |
| P1.2 | cross-lexeme nearest at `mention_end` | 0.60 |
| P1.3 | cross-lexeme nearest at `anaphor` (dimension attached to the referent) | 0.35 (clearly above control, below unit position) |
| P1.4 | **lattice holdout** nearest (held-out points, base points in train) at best position | 0.35; per-axis R² 0.6 |
| P1.5 | lattice holdout is better for *named* held-out points than for unnamed ones | yes, by ≥ 0.15 |
| P1.6 | additivity residual ‖c(d1d2)−c(d1)−c(d2)+c(1)‖ relative to random-triple baseline | 0.5 (i.e. half the baseline) |
| P1.7 | DIMLESS projection lands nearer the origin than any base point | yes for ≥ 70% of items |
| P1.8 | transfer REAL→INV-LEX (cross-condition) nearest at `mention_end` | 0.40 |
| P1.9 | transfer REAL→XLING | 0.55 |
| P1.10 | train neutral → test revealing (and reverse) | within 0.10 of within-family |
| P1.11 | noun_only (no unit) decodes the *revealing noun's* dimension at ≥ 0.5 nearest (base dims only) | yes |
| P1.12 | categorical (per-point logistic) beats linear-in-exponent on random splits by ≥ 0.10, but cannot extrapolate; MLP beats linear by ≤ 0.10 | yes |
| P1.13 | pre-answer position of T1v2 prompts: probe trained on mentions decodes the *composed* dimension at ≥ 0.30 nearest (transfer, no retraining) | 0.30 |
| P1.14 | larger models (14B, 32B) extrapolate better than 4B on lattice holdout | +0.10 |

## G1 decision rule (pre-registered)
**PASS (H1+H2 support)** if, for ≥ 2 base models: cross-lexeme selectivity (task − control, nearest) ≥ 0.30 at `mention_end` or `anaphor` **and** lattice-holdout nearest ≥ 0.25 with per-axis R² ≥ 0.4 at some (layer, position).
**PARTIAL (H1 only)** if cross-lexeme selectivity ≥ 0.30 somewhere but lattice holdout < 0.25 everywhere: dimension is decodable but not linearly lattice-structured → Phase 2 leads with adjudication (M_heur favoured), and the paper tilts to "dissociation / graded".
**FAIL** if cross-lexeme selectivity < 0.30 at every position beyond `unit`: dimension does not leave the lexeme → heuristics-extension branch; Phase 2 becomes a lexical-cue DAS study mirroring the rival paper.

## What would make me distrust a PASS
- Lattice extrapolation driven only by axis-wise sign (e.g. every L^a M^b T^c predicted with the right signs but wrong magnitudes): report magnitude-only and sign-only accuracies; PASS requires exact-point nearest ≥ 0.25.
- Control task not at chance (leaky lexeme grouping): verify control ≈ chance.
- Extrapolation only at the `unit` token (a lexeme-embedding artefact): PASS requires it at `mention_end`, `anaphor`, or `last`.

## What would make me distrust a FAIL
- Parameterisation: if the categorical probe is high but linear-in-exponent is low, test sign/magnitude and MLP before concluding; if any of these extrapolate, re-score.
- Position: if all mass sits at the last token of the passage rather than at mention_end/anaphor, add position classes (next-sentence tokens) before concluding.

## Calibration commitment
Score P1.1–P1.14 at G1 (hit = within ±0.15 for accuracies / right sign), and write what each miss implies.
