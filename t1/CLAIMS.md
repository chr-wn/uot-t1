# CLAIMS ledger

Status vocabulary: **supported** (survived kill attempts + controls), **tentative** (observed, not yet attacked), **weak**, **refuted**, **retracted**. Every entry lists the experiment IDs behind it. The final report may assert only ledger-backed claims at their ledger strength.

| ID | Claim | Status | Evidence (experiment IDs) | Kill attempts run | Notes |
|---|---|---|---|---|---|
| C1 | Six base LMs (4B–32B, three families) select the dimensionally correct composed unit for invented lexemes and unnamed lattice points at 0.84–0.97 (4-way candidate readout), with no surface-order penalty; invented cells within ±0.05 of familiar twins. | supported | E0.3 (T1 v2, 2 seeds), E0.3b | length-matching; prefix-free candidates; order swap; two seeds; corpus-verified lexemes | Competence claim only; see C2 |
| C2 | A large share of C1 (0.38–0.86 of the 4-way accuracy) is achievable with the scenario's unit tokens deleted, i.e. from relation wording + candidate structure (+ definitions); the full prompt adds +0.08 to +0.47. | supported | E0.3b | — | Caps the interpretive weight of C1; motivates Phase-1 positions |
| C3 | Free generation of the composed unit is dimensionally correct in 42–83% of generations that contain a unit; errors are dominated by dropped exponents/factors. | tentative | E0.3 (generation readout), A4 | none yet (readout coverage 0.7–0.96) | Needs the no-number free-generation twin (E1.0b) |
| C4 | Behavioural accuracy shows no gradient with lattice distance (≤ 8) or Dolma-1.7 frequency under the candidate readout. | supported (ceiling-limited) | E0.3, E0.4 | — | Absence of gradient at ceiling is not evidence against A2.5 |
| C5 | Base models rank dimensional consistency/convertibility well (AUROC 0.82–1.00 for invented units, 0.56–0.95 familiar) but map it to yes/no with a large constant bias. | supported | E0.3 (T2/T4, calibrated readouts) | — | Raw yes/no accuracies are not quotable |
| C6 | Unnamed lattice points are harder than named ones in formula selection (T3) by 0.10–0.28 for every model. | supported | E0.3 (T3) | position-calibrated readout | Only behavioural "distance" effect seen |
