# G0 gate report — Phase 0 behavioural go/no-go

*Status: DRAFT skeleton written 2026-09-13 while the last runs (Qwen3-8B instruct, OLMo-3-32B, T1 v2 re-runs) finish; numbers below are filled from `runs/E0.3/analysis/` and `figures/phase0/tables.md` when complete. Sections marked [FINAL] are updated last.*

## 1. What ran
- **Stimuli** (single generator, Pint-validated; `data/phase0/`): T1 unit-of-answer cloze (v1: 2×1440 items; v2: 2×1440 items with prefix-free candidate sets), T2 consistency judgment (1440), T3 formula selection (720), T4 conversion acceptability (1200), T5 error spotting (720). Conditions: familiar-named, familiar-unnamed lattice points, invented lexemes (corpus-verified ≤ 50 Dolma-1.7 occurrences), invented base dimensions, plus familiar twins for every invented item; ≥ 2 templates per relation in both surface orders; five surface styles.
- **Models**: Qwen3-4B/8B/14B-Base, OLMo-3-7B, OLMo-3-32B, Gemma-2-9B (base); Qwen3-4B, Qwen3-8B (instruct, thinking off). Run on rosetta18 A6000s; every run is `03_run_battery.py` with a fixed config; per-item candidate scores and greedy generations stored under `runs/E0.3/<model>/`.
- **Readouts**: `acc_cand` = argmax of mean per-token log-prob over 4 (T1/T3), 3 (T5) or 2 (T2/T4) candidates [primary, as pre-registered]; for 2-way tasks also AUROC of the yes–no margin and median-calibrated accuracy (added after observing constant yes/no biases; see §5); `acc_gen_dim|parsed` = fraction of greedy generations that contain a parseable unit expression whose *dimension* equals the answer's (secondary).
- **Controls** (E0.3b): the same T1 candidate sets scored with (a) no context (definitions + "The unit is"), (b) the scenario with all unit tokens deleted (relation words and definitions kept).
- **Frequency instrument** (E0.4): Dolma-1.7 infini-gram counts for all 23k unit strings used; invented answer strings have count ≤ 1.

## 2. Headline: compositional unit-of-answer (T1) [FINAL: table from tables.md]
(placeholder — filled at gate)

## 3. Controls and confounds (T1)
- Prior-only (no context) accuracy is at or below chance for familiar units and 0.3–0.45 for invented units (definitions alone bias toward common lattice points such as length/time).
- "No units in scenario" retains 0.42–0.76 (Qwen3-4B): the relation sentence + candidate structure (+ definitions, for invented units) support a verbal route to the answer that never binds the mentioned quantities. The full prompt adds +0.19 to +0.46. [FINAL: all models]
- Order-swapped items are not harder than in-order items for any model (Fig. 2): template copying is excluded as the explanation.
- Residual length bias after length-matching: correct candidate strictly shortest in 12% (v2), strictly longest in 1%. [FINAL: length-matched subset accuracy]
- Prefix candidates (v1 bug, area relation) removed in v2; v1 vs v2 comparison reported. [FINAL]

## 4. Secondary tasks (T2–T5) [FINAL: table]
(placeholder)

## 5. Anomalies (all logged in NOTEBOOK.md the day found)
A1 real words leaking into "invented" lexemes (fixed before any headline run); A2 area-relation dip = prefix candidates (fixed, v2); A3 constant yes/no bias in base models (readout added, not a stimulus change); A4 free generation often continues the output number instead of producing a unit (readout limitation; parse-conditional metric added).

## 6. Pre-registered criterion and outcome [FINAL]
(placeholder: (a) INV-LEX ≥ 0.50 with CI lower ≥ 0.40; (b) order-swapped INV-LEX ≥ 0.40 with CI lower ≥ 0.30; (c) FAM-UNNAMED ≥ 0.50 — per model)

## 7. Calibration against PREDICTIONS_phase0.md [FINAL]
(placeholder: P0.1–P0.11 hit/miss with implications)
