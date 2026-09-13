# DESIGN — Phase 0: foundations + behavioural go/no-go

Status: written 2026-09-12 before any model run. Adversarial pass recorded in `decisions/DR-001`.

## 0. Question and claim ladder (for reference)
Does a pretrained LM maintain an internal variable over the dimension lattice (free abelian group ℤ^k of exponent vectors) that it *uses* to license operations? Rungs: H1 decodability → H2 lattice geometry → H3 causal gating → H4 productive composition. Rivals: A0 lexical co-occurrence, A1 template/formula retrieval, A2 bag of heuristics, A2.5 graded algebra (structure near the origin decaying with lattice distance / frequency), A3 the algebra.

## 1. Stimulus library (`src/uot`)

### 1.1 Dimension algebra (`uot.dims`)
- `Dimension`: immutable map base-symbol → integer exponent over an ordered base set. Real base set = SI-7 {L, M, T, I, Θ, N, J}; invented base symbols (`X1`, `X2`, …) can be appended per stimulus context. Operations: `*`, `/`, `**n`, equality, `distance()` = L1 norm of the exponent vector (lattice distance from the origin), `is_dimensionless`.
- Ground truth for real units comes from Pint (`pint.UnitRegistry().get_dimensionality`), mapped onto the SI-7 basis; every registry entry is validated against Pint in tests.

### 1.2 Unit registry (`uot.units`)
Each `Unit` has: canonical id; `Dimension`; scale to SI base (Pint); `system` ∈ {SI, SI-prefixed, cgs, imperial/US, archaic, nautical, astronomical, named-derived, invented}; surface forms: short symbol(s) ("km"), long singular/plural ("kilometer"/"kilometers"), and translations for cross-lingual transfer (es, de, fr, where they exist); flags: `named_derived` (N, J, W, Pa, Hz, …), `invented`.
- Familiar coverage (target ≥ 60 real units): length (mm cm m km in ft yd mi nmi furlong league fathom parsec light-year cubit rod chain Å), mass (mg g kg t oz lb stone grain slug tonne), time (ms s min h d week fortnight yr), current (A mA), temperature (K, °C used only in Δ contexts), amount (mol), luminous intensity (cd), plus named derived: N dyne lbf kgf, J erg cal kWh eV, W hp, Pa psi bar atm, Hz Bq, C, V, Ω, m/s knot, L gal, ha acre.
- **Near-miss sets** (for judgment tasks): same-dimension-different-name {J, N·m, erg, cal, kWh, eV}, {Hz, rad/s, Bq, s⁻¹}, {Pa, psi, N/m²}, {Gy, Sv}; differ-by-one-step {kW vs kWh (T)}, {lb vs lbf (L T⁻²)}, {kg vs kgf}, {W vs J (T⁻¹)}, {m vs m² vs m³}, {m/s vs m/s²}, {Pa vs N (L⁻²)}, {C vs A (T)}.
- **Invented lexemes**: pronounceable CVC/CVCC nonsense words (generated, filtered against an English wordlist and Pint's name space; e.g. blork, zim, vurn, plath, dreb). A stimulus context defines each invented unit exactly once ("A blork is a unit of length."). Invented lexemes are *rotated* across dimensions across items so no lexeme is ever tied to one dimension globally (kills A0 at the lexeme level).
- **Invented base dimensions**: "Flarn is a basic physical quantity, measured in glorps." The lexeme for the quantity and the unit are both fresh; compositions then use glorps with real units (glorps per second, glorp·meters).
- **Adversarial lexemes** (TypeProbe-style): invented units whose form resembles a real unit of a *different* dimension (e.g. "kilomet" defined as a unit of mass), used in a small robustness cell, never in the headline cell.

### 1.3 Quantities and surface forms (`uot.quantity`)
`Quantity(value, unit)` renders as numeral + unit with sampled surface options: numeral format (integer / 1–2 decimal places), unit form (short / long / long-plural), separator ("5 m", "5m", "5 meters"), compound-unit notation ("km/h", "km per h", "kilometers per hour", "m^2", "m²", "square meters", "kg·m/s²", "kg m/s^2"). The *same* generator produces every task, so ground truth (dimension, scale, value) is single-sourced.

### 1.4 Task families (`uot.tasks`), all template-distributional
Every task instance carries: `item_id`, `task`, `condition`, `template_id`, `prompt`, `candidates`, `answer_index`, `answer_dimension`, `lattice_distance`, `unit_strings` (for frequency lookup), and per-slot `Quantity` metadata.

- **T1 UnitCloze** (headline for G0): a scenario states two or three input quantities and names the derived quantity; the model completes the unit. Relations (ground truth via dimension arithmetic): speed L/T, acceleration L/T², area L², volume L³, density M/L³, force M L/T², energy M L²/T², power M L²/T³, pressure M/(L T²), momentum M L/T, frequency 1/T, flow L³/T, and **unnamed** points {M² L, L³/T⁵, M/T², L·T, M L³, T²/L, M² /T}. Inputs may be familiar or invented; the expected answer is the *composed* unit expression built from the input lexemes (e.g. "blorks per zim", "vurn·plath²/zim²").
  - Candidate set K=4: correct; template-copy distractor (input units in *surface order* joined by "per"/"·" — equals the correct answer only when the surface order matches the composition order, so half the items are **order-swapped** so that copying yields the wrong answer); inverted composition; a single input unit (non-composed). Readout = argmax mean-token-logprob over candidates; greedy-generation exact match reported as secondary.
  - Conditions: FAM-NAMED (km/h), FAM-UNNAMED (kg²·m), INV-LEX (invented lexemes, real dimensions, defined in context), INV-BASE (invented base dimension), each INV item paired with a FAM *twin* sharing template, numbers and relation.
- **T2 ConsistencyJudgment**: "Does it make sense to add 5 m and 3 s?" / "…to compare 5 J with 3 N·m?" / "…to set 5 kg equal to 3 lbf?" — Yes/No with balanced labels; includes near-miss pairs and invented-unit twins. Operations: add, compare, equate, convert.
- **T3 FormulaSelection**: "Which expression has the units of pressure (Pa)? (A) F/A (B) F·A (C) A/F (D) F/A²" with symbols defined in-context (including invented-unit versions).
- **T4 ConversionAcceptability**: "Can a quantity in kilowatt-hours be converted into kilowatts? Yes/No" over near-miss pairs and lattice-distance-graded pairs.
- **T5 ErrorSpotting**: a 3-sentence passage with three quantities, one dimensionally wrong ("The car's mass is 1200 m."); "Which sentence contains a unit error? (A/B/C)".

Instruct models get the same items wrapped in the chat template with an "answer only with …" instruction; Qwen3 instruct is run with thinking disabled. Base and instruct results are reported separately and never pooled.

### 1.5 Nuisance controls built into every task
- Numerals sampled independently of the correct answer; values kept in [1, 999] to avoid tokenisation cliffs; dimension never predictable from the numeral.
- Surface-order swap (T1) and yes/no balance (T2, T4) so no cue-free heuristic exceeds chance.
- Template distributions: ≥ 6 templates per task; items sampled so each (condition × template) cell is populated; CIs are bootstrapped over items *clustered by template* and over seeds.
- Invented ↔ familiar twins (requirement 6) in every task.

## 2. Behavioural battery (E0.3)
Models (base): Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base, Olmo-3-1025-7B, Olmo-3-1125-32B, Mistral-7B-v0.3 (third family; see F1). Instruct: Qwen3-4B, Qwen3-8B. Cells: task × condition × template; N = 240 items per (task, condition) (≈ 40 per template), sampled with a fixed seed; two seeds of stimulus sampling for the headline T1 cell.
Analyses (E0.5): accuracy by condition (with twin deltas); accuracy vs. lattice distance (per condition); accuracy vs. log corpus frequency of the answer unit string (infini-gram, Dolma-1.7; FAM only — INV strings have zero count by construction, which we verify); ordinal regressions with both predictors; order-swap penalty; base-vs-instruct comparison.

## 3. Readout definitions
- `acc_cand`: argmax over the K candidate strings of mean per-token log-prob of the candidate continuation (rival paper's readout). Primary.
- `acc_gen`: greedy generation (≤ 12 tokens) normalised and matched to the correct unit expression (set of acceptable renderings). Secondary.
- `margin`: logprob(correct) − max logprob(distractor). Used for surrogate analyses.
- Instruct: same, over the option strings after the assistant turn starts.

## 4. Corpus-frequency instrument (E0.4)
infini-gram `v4_dolma-v1_7_llama` counts for every unit string used (short/long/plural forms, compound renderings); feature = log(1 + max count over renderings). Documented as a proxy for OLMo-3's corpus (F2). Invented strings must return 0 (sanity check).

## 5. G0 criterion
Pre-registered in `PREDICTIONS_phase0.md` (thresholds, decision rules, and what would reverse them). Shape: at least one base model well above chance on compositional unit-of-answer with *novel combinations* (INV-LEX and FAM-UNNAMED), where "well above chance" survives the order-swap control.

## 6. Amendments to the brief (with reasons)
1. Third family: Mistral-7B-v0.3 instead of Gemma-2-9B/Llama-3.1-8B (gated on this account; F1). Gemma Scope SAE analysis becomes conditional on access.
2. Frequency instrument: Dolma-1.7 infini-gram index (no OLMo-3 index exists); claims about "training-corpus support" are phrased as Dolma-1.7 support and treated as a proxy (F2).
3. G0 criterion refined: "novel combinations" must be tested with surface-order-swapped items and a template-copy distractor in the candidate set; otherwise a model that copies "A in B → A per B" passes G0 without any dimension variable (DR-001).
4. Position classes for Phase 1 include the *pre-answer* token of a cloze (where the derived dimension must exist if the model composes), not only unit/numeral/end-of-mention/anaphor.
5. Value range restricted to [1, 999] integers/short decimals (tokenisation stability) — narrower than the rival paper's 1e-3…1e4 because scale is not our variable.

## 7. Phase-0 experiment list
| ID | What | Output |
|---|---|---|
| E0.0 | env build + model/library smoke test + unit-string tokenisation audit | `runs/E0.0/` |
| E0.1 | registry/dimension library unit tests vs Pint | `tests/` green |
| E0.2 | battery debug on Qwen3-4B-Base (small N) | `runs/E0.2/` |
| E0.3 | full battery, all models | `runs/E0.3/` |
| E0.4 | infini-gram frequency table | `data/frequency/` |
| E0.5 | analyses + figures | `figures/phase0/`, `reports/G0_gate_report.md` |
