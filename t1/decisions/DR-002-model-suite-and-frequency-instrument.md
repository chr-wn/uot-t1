# DR-002 — Deviations from the brief's engineering defaults (2026-09-12)

## 1. Third model family
- Brief: Gemma-2-9B (Gemma Scope SAEs) or Llama-3.1-8B.
- Observed: both are gated for the HF account whose token is on this machine (403 on `config.json` with the token; `whoami` succeeds). No other credentials available; using third-party mirrors of gated weights is not acceptable without the license acceptance.
- Decision: third family = **Mistral-7B-v0.3** (ungated; different pretraining corpus and tokenizer from Qwen3 and OLMo-3). Gemma-2-9B is added if access is granted (FLAG F1). Cost: no SAE-based latent reconnaissance in Phase 1 unless Gemma arrives; substitute = probe-derived subspaces only.
- Reversal condition: PI accepts the Gemma license → Gemma-2-9B joins the suite in Phase 1.

## 2. Corpus-frequency instrument
- Brief: OLMo + infini-gram, verify index coverage for the OLMo version used.
- Observed: infini-gram serves `v4_dolma-v1_7_llama` (Dolma 1.7, the OLMo-1.7 pretraining corpus) and Pile/C4/RedPajama indices; no Dolma-3/OLMo-3 or OLMo-2 index is served under any name tried.
- Decision: use Dolma-1.7 counts as a **proxy** for pretraining-corpus support of unit strings. Justification: Dolma 1.7 and Dolma 3 share the bulk web/scientific sources; unit-string frequencies are dominated by web text and should be rank-stable across such corpora. Every frequency claim in reports is phrased as "Dolma-1.7 frequency". Cost: "zero training-corpus support" cannot be asserted for OLMo-3 — we assert "zero Dolma-1.7 support" and treat it as strong but indirect evidence.
- Reversal condition: an OLMo-3 index appears → recompute the frequency table (one API sweep, cheap).

## 3. Torch build
- rosetta4 driver 550 / CUDA 12.4 → torch 2.6.0+cu124 (cu126/cu128 builds fail to initialise CUDA here).

## 4. Value range
- Rival paper: 1e-3 ≤ r < 1e4 with 3 decimals. We restrict to integers / ≤ 2 decimals in [1, 999]: scale is not our variable and long decimals add tokenisation noise at the numeral position that Phase 1 probes at.
