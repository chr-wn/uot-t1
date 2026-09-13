# G2 gate report — Phase 2 causal adjudication (draft 2026-09-13; [FINAL] cells filled when the seed replicates, controls and OLMo/Gemma replication complete)

Companions: `reports/G2_red_team.md`, `decisions/DR-005-G2-decision.md`, `PREDICTIONS_phase2.md`, `figures/phase2/`, `runs/E2.*`.

## 1. What ran
- **Task and stimuli.** Two-quantity yes/no consistency judgments (add / compare / equate / convert; 2 templates each), single-token unit surface forms (43 lexemes over 10 lattice points, audited on the tokeniser), values 2–99. Interchange examples: 4,800 real (base/source pairs, balanced so that half the sources share quantity 2's dimension), 1,600 invented-unit twins (dimension defined in context), train/eval split by base item; flagged subsets: *disagree* (M_alg and M_heur labels differ), *same-dimension lexeme swap*, *held-out lexemes* (never in DAS training).
- **Rival high-level models.** M_alg: D1 (categorical dimension of quantity 1), answer = yes iff D1 = D2. M_heur: U1 (unit lexeme), answer = the model's own pairwise table (52×52 lexemes × 4 ops, 16k prompts; AUROC vs dimension equality 0.89–0.98; rank-calibrated threshold at the true same-dimension rate). Counterfactual labels are computed by each model, never by hand.
- **Interventions** (own implementation, unit-tested): residual-stream interchange at the unit-1 token (u1), unit-2 token (u2) and answer token, layers 4–28; learned orthogonal subspaces of rank 1–128 (Adam on an orthogonal parametrisation, 1,000 steps, lr 5e-3), matched-rank random subspaces, probe-derived subspaces (Phase-1 logistic-16 and ridge-3 directions), and the full-residual patching ceiling. **Balanced IIA** = mean of IIA on output-changing and output-preserving examples relative to the model's *own* base prediction (chance 0.50).
- **Controls** [FINAL]: 3 seeds; natural-distribution projection (Makelov); value-preservation probe; binding (D2 readout after a D1 swap); noising; dose–response; text-rewrite baseline.

## 2. Where quantity 1's unit is causally read (ceiling)
Replacing the *entire* residual of the u1 token: layer 4 → balanced IIA 0.67 under M_alg labels (40% lawful flips) and 0.85 under M_heur labels (73%); layer 8 → 0.65 / 0.78; layer 12 → 0.57 / 0.63; layer ≥ 16 → 0.50 (no effect). u2: 0.52–0.55 / 0.63–0.66 at layers 4–16. Answer token: nothing until layers 24–28, where a late decision variable flips 28–48% (this is the answer, not D1). Invented units: 0.50 at every site — their dimension is carried by the definition sentence, not the lexeme token. **The model's judgments follow its pairwise table far more than dimension equality even under full patching** (0.73 vs 0.40 flips): the equality model is a worse description of the behaviour than the lexical table, before any subspace question arises.

## 3. Headline: matched-rank subspaces at u1 (Qwen3-4B-Base; Fig. P2a)
| rank | M_alg learned (L4 / L8) | M_heur learned (L4 / L8) | random | probe-derived (L8): alg / heur |
|---|---|---|---|---|
| 1 | 0.50 / 0.50 | 0.51 / 0.52 | 0.50 | — |
| 3 (ridge exponent basis) | — | — | — | 0.50 / 0.51 |
| 4 | 0.51 / 0.51 | 0.53 / 0.53 | 0.50 | — |
| 16 | 0.53 / 0.57 | 0.58 / 0.60 | 0.50 | 0.54 / 0.57 |
| 64 | 0.58 / 0.60 | 0.61 / 0.64 | 0.50 | — |
| 128 | 0.63 / 0.61 | 0.70 / 0.65 | 0.50 | — |
| full (2560) | 0.67 / 0.65 | 0.85 / 0.78 | — | — |
Subsets at rank 64, L8: disagree 0.48 (alg) vs 0.54 (heur); held-out lexemes 0.59 / 0.66 (no memorisation of lexemes); invented 0.50 / 0.50; the alg-trained subspace scored with heur labels: 0.66 (> its own 0.60). Training accuracy of the alg subspace never exceeds 0.62 (heur 0.78). [FINAL: seed means ± sd]

## 4. Controls [FINAL]
(natural-projection separation; value R² before/after; D2 accuracy before/after; noising flip rate vs random; dose–response; text-rewrite reference)

## 5. Replication (OLMo-3-7B, Gemma-2-9B) [FINAL]

## 6. Composition (H4, exploratory, E2.5)
Implanting slot 2's dimension into slot 1 of a composition prompt (full residual, u1, L4/L8): the correct composed candidate loses 1.7–1.8 nats (text rewrite −1.9), argmax flips 7–14% (rewrite 18%), matched-rank random 0.0 nats — the composed answer depends causally on slot 1's unit representation, but no lawfully composed alternative was in the candidate set, so lawful propagation is not scored (n = 44).

## 7. Pre-registered decision rules — evaluation [FINAL]
- Algebra/categorical-causal: requires M_alg ≥ 0.70 with invented ≥ 0.65 and held-out ≥ 0.65 and alg ≥ heur on disagree — **not met** (best alg 0.63; invented 0.50; disagree alg < heur).
- Heuristics-extension: M_heur ≥ M_alg + 0.10 on disagree with M_heur ≥ 0.65 and alg invented ≤ 0.55 — disagree gap 0.06–0.14 at rank 64–128, M_heur on disagree 0.54–0.63 (< 0.65): **borderline, not fully met**.
- Dissociation: ceiling ≥ 0.75 but neither subspace > random + 0.10 at rank ≤ 64 — **not met** (heur 0.64 = random + 0.14 at rank 64).
- Graded: alg IIA ≥ 0.70 for base but ≤ 0.55 for derived — [FINAL: per-point analysis from 3-seed records].
Provisional reading: none of the four pre-registered branches is met cleanly; the evidence sits between heuristics-extension and graded, and the honest branch is stated in DR-005 with the rule text quoted.
