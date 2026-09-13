# G2 red-team appendix (2026-09-13)

Reading under attack: *"the causally effective unit representation behaves like a lexical pairwise heuristic, not a dimension-equality variable."*

## Attack 1 — The equality model was handicapped by the task
The model's base accuracy on these yes/no prompts is 0.69 (Qwen) / 0.61 (OLMo) with a strong "no" bias; the ceiling under equality labels is therefore ≤ 0.67 by construction, and the DAS objective under equality labels is partly unlearnable (train accuracy 0.59). **Partly conceded.** Mitigations already in the report: ceiling-normalised IIA (alg reaches 92% of its ceiling); per-operation results (convert, the best-calibrated op, reaches 0.70). Not excluded: with a better-calibrated task (instruct models with a generation readout, or few-shot prompting) the equality ceiling could rise and the subspace comparison could change. Cheapest follow-up: repeat at the `convert` op only, or with 3 in-context examples.

## Attack 2 — Rank inflation: M_heur wins because lexical identity is higher-dimensional
The lexical table has more free parameters than equality, and DAS with a larger effective target may fit better at any rank. **Partly excluded:** at matched rank the gap is small (0.02–0.05) and *both* keep rising with rank; the decisive comparisons are label-based on the same subspace (alg-trained subspace scored with heur labels: 0.66 > 0.60) and the disagree subset, which do not depend on capacity.

## Attack 3 — The invented-unit null is a site artefact
Invented units' dimension is in the definition sentence; patching the lexeme token cannot move it, so "no effect for invented units" says nothing about whether a dimension variable exists elsewhere (e.g. at the referent or definition tokens). **Conceded.** The claim in the report is limited to the lexeme token; a definition-token/referent-site DAS is the top item in NEXT.md.

## Attack 4 — Value contamination shows the swap is too coarse, not that the variable is absent
A rank-64 swap moves the bound number–unit representation; a smaller, purer dimension subspace might exist inside it. **Not excluded**, but the rank sweep shows ranks ≤ 16 have little causal effect (≤ 0.57), so any pure low-rank dimension subspace is causally weak at this site.

## Attack 5 — Only one model has the full battery
OLMo-7B replicates the pattern at one seed without controls; Gemma-2-9B could not be adjudicated (base accuracy at chance; bias-corrected re-run pending). **Conceded**; generality claims are limited to Qwen3-4B + OLMo-7B.

## Verdict
The heuristics-extension reading stands as the best-supported branch, with the algebra steelman preserved as "imperfect categorical use". The strongest open threats are Attack 3 (site) and Attack 1 (task calibration); both are cheap and listed first in NEXT.md.
