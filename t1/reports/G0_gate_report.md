# G0 gate report — Phase 0 behavioural go/no-go (2026-09-13)

Companion documents: `reports/G0_red_team.md` (adversarial pass), `decisions/DR-003-G0-decision.md` (decision record), `figures/phase0/` (Figs 1–5, tables.md), `runs/E0.3/analysis/` (CSV), `PREDICTIONS_phase0.md` (pre-registered).

## 1. What ran
- **Stimuli** (one Pint-validated generator, `src/uot`): T1 unit-of-answer cloze — v1 2×1440 items and **v2 2×1440 items with prefix-free candidate sets** (headline uses v2; v1 retained); T2 consistency judgment (1440); T3 formula selection (720); T4 conversion acceptability (1200); T5 error spotting (720). Conditions: familiar-named lattice points, familiar-unnamed points (kg²·m, L³/T⁵, …), invented lexemes (Dolma-1.7 count ≤ 50 per lexeme, ≤ 1 for any composed answer string), invented base dimensions, and a familiar twin for every invented item. Every relation has ≥ 2 templates in both surface orders; five surface styles; values 2–60.
- **Models** (rosetta18, A6000): base — Qwen3-4B/8B/14B, OLMo-3-7B/32B, Gemma-2-9B; instruct — Qwen3-4B/8B (thinking off). Per-item candidate scores and greedy generations are stored under `runs/E0.3/<model>/` (instruct re-runs with a generation readout: `runs/E0.3i/`, see §6).
- **Readouts.** `acc_cand` (pre-registered primary): argmax of mean per-token log-prob over K candidates (K=4 T1/T3, 3 T5, 2 T2/T4). Added during the phase, without changing stimuli: `acc_gen_dim|parsed` (greedy generation contains a unit expression whose parsed *dimension* equals the answer's); for two-way tasks AUROC of the yes–no margin and median-calibrated accuracy; for T3/T5 per-position-median calibrated accuracy. 95% CIs: bootstrap over templates then items.
- **Controls** (E0.3b): identical candidate sets scored with (a) no context (definitions + "The unit is") and (b) the scenario with every unit token deleted (relation words and definitions kept).
- **Frequency instrument** (E0.4): Dolma-1.7 infini-gram counts for 23k unit strings.

## 2. Headline — compositional unit-of-answer, T1 v2 (chance 0.25)

| model | FAM-NAMED | FAM-UNNAMED | INV-LEX | INV-LEX twin | INV-BASE | INV-BASE twin |
|---|---|---|---|---|---|---|
| Qwen3-4B-Base | 0.98 [0.96, 1.00] | 0.85 [0.77, 0.91] | 0.95 [0.91, 0.98] | 0.92 | 0.94 [0.89, 0.99] | 0.88 |
| Qwen3-8B-Base | 0.99 | 0.84 [0.74, 0.92] | 0.96 [0.92, 0.98] | 0.91 | 0.91 [0.83, 0.99] | 0.94 |
| Qwen3-14B-Base | 0.98 | 0.88 [0.81, 0.94] | 0.97 [0.93, 0.99] | 0.93 | 0.94 [0.87, 0.99] | 0.94 |
| OLMo-3-7B | 0.97 | 0.86 [0.80, 0.91] | 0.91 [0.87, 0.95] | 0.92 | 0.88 [0.77, 0.98] | 0.90 |
| OLMo-3-32B | 0.98 | 0.91 [0.84, 0.96] | 0.95 [0.91, 0.98] | 0.94 | 0.91 [0.85, 0.97] | 0.94 |
| Gemma-2-9B | 0.98 | 0.87 [0.78, 0.93] | 0.89 [0.84, 0.94] | 0.92 | 0.90 [0.82, 0.98] | 0.90 |

n = 480 per cell (two stimulus seeds). Order-swapped items are never worse than in-order items (swap penalty −0.09 to +0.05; Fig. 2). Length-matched subset accuracy equals the full-cell accuracy within 0.01 in every cell. Invented-unit cells are within ±0.05 of their familiar twins. Dimension-correct free generation (among generations that contain a unit): 0.75–0.83 (FAM-NAMED), 0.62–0.83 (FAM-UNNAMED), 0.42–0.82 (INV-LEX), 0.38–0.48 (INV-BASE) — free generation is a harder and noisier readout (§5, A4).

**Lattice distance and corpus frequency (v2).** Within INV-LEX, Spearman ρ(accuracy, L1 distance) is −0.15 (Qwen3-14B, p=0.001), −0.10 (OLMo-32B, p=0.02) and not significant for the other models; within FAM cells ρ is slightly *positive* (+0.03 to +0.11). Accuracy vs Dolma-1.7 log-count of the answer string (FAM items) is flat (0.89–0.97 across bins). There is **no behavioural gradient with lattice distance or frequency** at distances ≤ 8 under the candidate readout.

## 3. Controls and confounds (T1 v2)

| model | no context | no units in scenario | full |
|---|---|---|---|
| Qwen3-4B-Base (INV-LEX / FAM-UNNAMED) | 0.45 / 0.05 | 0.74 / 0.38 | 0.95 / 0.85 |
| Qwen3-14B-Base | 0.42 / 0.06 | 0.80 / 0.52 | 0.96 / 0.88 |
| OLMo-3-7B | 0.42 / 0.07 | 0.64 / 0.46 | 0.91 / 0.86 |
| OLMo-3-32B | 0.41 / 0.06 | 0.81 / 0.48 | 0.95 / 0.91 |
| Gemma-2-9B | 0.43 / 0.11 | 0.66 / 0.50 | 0.89 / 0.86 |

(full table: `figures/phase0/prior_controls.csv`; Fig. 3.) The candidate prior alone is at or below chance for real units; with invented lexemes the definitions alone bias the choice toward common lattice points (0.41–0.46). Deleting the unit tokens from the scenario but keeping the relation sentence and the definitions retains 0.38–0.86: a *verbal* route (relation words → exponent pattern over the available lexemes) accounts for a large share of candidate accuracy. The full prompt adds +0.08 to +0.47 (median ≈ +0.30), most for FAM-UNNAMED and the twins (where no definitions exist).

## 4. Secondary tasks (base models; raw accuracy / bias-free readouts)
- **T3 formula selection** (chance 0.25): named 0.77–0.99, unnamed 0.53–0.90, invented 0.60–0.96; Qwen3 > OLMo/Gemma; unnamed always hardest.
- **T2 consistency** (chance 0.5): raw 0.50–0.88 with a constant yes/no bias (Gemma "yes" 75–100%, Qwen/OLMo "no" 70–99%). AUROC of the margin: invented units 0.82–0.97 (Qwen/OLMo), 0.65–0.69 (Gemma); familiar pairs 0.56–0.95, lowest for near-miss pairs (0.47–0.70).
- **T4 conversion** (chance 0.5): raw 0.67–0.92 (Qwen/OLMo), ≈0.5 raw but AUROC 0.67–0.77 (Gemma); invented AUROC 0.95–1.00.
- **T5 error spotting** (chance 0.33): letter-position bias (option B rarely chosen); calibrated accuracy 0.30–0.65; only Qwen3-14B is clearly above chance in every cell (0.60–0.64).
Full table: `figures/phase0/tables.md`, Fig. 5.

## 5. Anomalies (logged in NOTEBOOK.md the day found; none reinterpreted after the fact)
A1 real words leaking into "invented" lexemes (fixed before any headline run; pool re-verified against Dolma-1.7). A2 area-relation dip (0.36–0.39) = prefix candidates favoured by per-token readout → v2 stimuli; v1 vs v2 differs only in FAM-NAMED (+0.04 to +0.06). A3 constant yes/no bias (readout added). A4 free generation often continues the output number or closes the sentence instead of producing a unit (readout coverage 0.7–0.96; conditional metric reported). A5 letter-position bias in T5/T3 (position-calibrated readout). A6 Qwen3-8B-instruct does not comply with "only the letter/unit" and is format-confounded under candidate scoring; re-run with a generation readout (§6).

## 6. Instruct models
Qwen3-4B-instruct (candidate readout): T1 0.64–0.89, T2 raw 0.57–0.84 (less bias than base), T4 0.75–0.87, T3 0.86–0.96, T5 0.47–0.55. Qwen3-8B-instruct under candidate scoring is format-confounded (A6). **Re-run with candidates without leading space and a 24-token generation readout (`runs/E0.3i/`, 2026-09-13 11:30).** T1 candidate readout: Qwen3-4B 0.70–0.90, Qwen3-8B 0.59–0.88 (INV-LEX 0.86 / 0.69) — lower than the base models, with unnamed points hardest (0.70 / 0.59). T2/T4 with the generation readout (coverage 0.86–1.00): T2 invented 0.73–0.85, familiar 0.54–0.88 (near-miss 0.60–0.62); T4 0.78–0.92. T3: Qwen3-4B 0.84–0.97; Qwen3-8B still does not emit a bare letter within 24 tokens (coverage 0.00–0.38) — its T3 numbers remain unreported. T5 generation readout (coverage 0.75–1.00): Qwen3-4B 0.50–0.61, Qwen3-8B 0.60–0.83. Conclusion: instruct tuning does not improve composition; it mainly removes the yes/no bias. None of this bears on the gate decision (criterion over base models).

## 7. Pre-registered criterion and outcome
For **every** base model: (a) INV-LEX ≥ 0.50 with CI lower ≥ 0.40 — observed 0.89–0.97, CI lower ≥ 0.84 ✔; (b) order-swapped INV-LEX ≥ 0.40 with CI lower ≥ 0.30 — observed 0.89–0.95 ✔; (c) FAM-UNNAMED ≥ 0.50 — observed 0.84–0.91 ✔. Distrust clauses: swap penalty > 0.30 — no (≤ 0.05); length-matched subset below threshold — no; acc_gen ≈ chance — no (dimension-correct generation 0.42–0.83 among parsed). **Outcome: PASS** (all six base models).

## 8. Calibration against PREDICTIONS_phase0.md (hit = within ±0.10 / right sign)
| # | predicted (4B / 14B) | observed | verdict |
|---|---|---|---|
| P0.1 FAM-NAMED | 0.85 / 0.92 | 0.98 / 0.98 | miss / hit |
| P0.2 FAM-UNNAMED | 0.60 / 0.75 | 0.85 / 0.88 | miss / miss |
| P0.3 INV-LEX | 0.60 / 0.78 | 0.95 / 0.97 | miss / miss |
| P0.4 INV-BASE | 0.55 / 0.72 | 0.94 / 0.94 | miss / miss |
| P0.5 swap penalty ≤ 0.15 / ≤ 0.10 | — | ≈ 0 | hit / hit |
| P0.6 T2 familiar / near-miss (raw) | 0.80/0.60 ; 0.90/0.70 | 0.54–0.69/0.60 ; 0.62–0.88/0.66 | miss / partial |
| P0.7 T4 near-miss | 0.60 / 0.72 | 0.77 / 0.71 | miss / hit |
| P0.8 ρ(acc, distance) in INV-LEX −0.3…−0.6 | — | −0.06 / −0.15 | miss (right sign, far weaker) |
| P0.9 frequency > distance in FAM | yes | neither has an effect | miss |
| P0.10 instruct ≥ base on T2/T4, ≈ base on T1 | yes | T2/T4 yes; T1 lower | partial |
| P0.11 acc_gen ≥ 0.10 below acc_cand on INV | yes | yes (0.1–0.5 below) | hit |
**What the misses imply.** I under-predicted compositional competence by 0.2–0.4 in every T1 cell and expected graded degradation with lattice distance/frequency that does not exist at this range under a candidate readout. The framing update: the behavioural bar of G0 is not discriminating between the algebra and a verbal composition route (§3), so Phase 1–2 carry the adjudication; the "graded boundary" (A2.5), if it exists, will not be found behaviourally with 4-way candidates at |e| ≤ 8 — it must be looked for in representations (extrapolation to held-out points) and in free generation, where exponent errors concentrate (A4). Judgment tasks, not composition, are where models are weak, and there the weakness is partly readout bias.
