# PREDICTIONS — Phase 0 (committed 2026-09-12, before any battery run; never edited after)

Notation: `acc_cand` = argmax over K=4 candidates by mean per-token log-prob; chance = 0.25 for T1/T3/T5 (K=4 / 4 options / 3 options → 0.33 for T5), 0.50 for T2/T4.

## Expected outcomes (point predictions, base models unless stated)
| # | Prediction | Qwen3-4B-Base | Qwen3-14B-Base / Olmo-3-32B |
|---|---|---|---|
| P0.1 | T1 FAM-NAMED acc_cand | 0.85 | 0.92 |
| P0.2 | T1 FAM-UNNAMED acc_cand | 0.60 | 0.75 |
| P0.3 | T1 INV-LEX acc_cand | 0.60 | 0.78 |
| P0.4 | T1 INV-BASE acc_cand | 0.55 | 0.72 |
| P0.5 | Order-swap penalty (acc[in-order] − acc[swapped]) in INV-LEX | ≤ 0.15 | ≤ 0.10 |
| P0.6 | T2 consistency, familiar pairs / near-miss pairs | 0.80 / 0.60 | 0.90 / 0.70 |
| P0.7 | T4 conversion acceptability, near-miss pairs | 0.60 | 0.72 |
| P0.8 | Accuracy vs lattice distance within INV-LEX: Spearman ρ | −0.3 to −0.6 | −0.2 to −0.5 |
| P0.9 | Within FAM, log-frequency explains more T1 variance than distance (partial R²) | yes | yes |
| P0.10 | Instruct (Qwen3-4B/8B, no thinking) ≥ base on T2/T4, ≈ base on T1 | yes | — |
| P0.11 | Generation exact-match acc_gen is ≥ 0.1 below acc_cand on INV cells (formatting variance) | yes | yes |

## G0 decision rule (pre-registered)
**PASS** if, for at least one base model:
- (a) T1 INV-LEX `acc_cand` ≥ 0.50 with bootstrap 95% CI lower bound ≥ 0.40 (chance 0.25), **and**
- (b) on the order-swapped INV-LEX subset alone, `acc_cand` ≥ 0.40 with CI lower bound ≥ 0.30, **and**
- (c) T1 FAM-UNNAMED `acc_cand` ≥ 0.50.

**PARTIAL** if (c) holds for some model but (a)/(b) fail for all: proceed to Phase 1 with invented-unit conditions demoted to secondary; H4-with-invented-units is pre-declared unlikely and the paper branch tilts toward graded algebra / heuristics.

**FAIL** if no base model reaches T1 INV-LEX ≥ 0.40 *and* no base model reaches FAM-UNNAMED ≥ 0.40: pivot to the heuristics-extension branch (Phase 1–2 still run on FAM-NAMED, reframed with the null favoured). If additionally FAM-NAMED < 0.5 everywhere, terminate to a negative-results report.

## What would make me distrust a PASS
- Order-swap penalty > 0.30 (template copying) even if (a) holds → treat as PARTIAL and say so.
- Candidate-length confound: if the correct candidate is systematically the shortest/longest and a length-matched subset drops below the threshold → PARTIAL.
- acc_gen ≈ chance while acc_cand passes → report both, treat readout as tentative, and add a generation-based re-check in Phase 1.

## What would make me distrust a FAIL
- Tokenisation pathologies for invented lexemes (e.g., lexemes split into >4 tokens) — verified in E0.0 before the battery; if found, lexeme generator is fixed *before* E0.3 (allowed: this is a stimulus-validity fix, not a threshold change).
- Prompt-format brittleness: if accuracy varies by > 0.3 across templates, the battery is re-run with the template distribution re-balanced and *both* runs are reported.

## Calibration commitment
At G0 I will score each of P0.1–P0.11 as hit / miss (miss = outside ±0.10 for accuracies, wrong sign for correlations) and write what each miss implies for the framing.
