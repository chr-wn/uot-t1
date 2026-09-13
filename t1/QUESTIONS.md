# QUESTIONS (would have asked the PI) → assumption adopted

| # | Date | Question | Assumption adopted |
|---|---|---|---|
| Q1 | 2026-09-12 | Gemma-2-9B / Llama-3.1-8B are gated for the HF account; accept licenses? | Use Mistral-7B-v0.3 as third family; add Gemma if access appears (F1). |
| Q2 | 2026-09-12 | No infini-gram index for OLMo-3's corpus; acceptable to use Dolma-1.7 as the frequency proxy? | Yes, with explicit proxy wording in every frequency claim (F2). |
| Q3 | 2026-09-12 | Which node/GPUs may I use? | rosetta4 A100s when idle (checked before each launch); fall back to rosetta11 A6000s. Never more than 4 GPUs at once without an explicit OK. |
| Q4 | 2026-09-12 | Storage budget for residual caches? | ≤ 1.5 TB under `t1/cache/` (9.9 TB free on NFS); caches store selected positions only, fp16. |
| Q5 | 2026-09-12 | Instruct models: allow "thinking" mode? | No — thinking disabled (`enable_thinking=False`) for constrained judgments; CoT experiments (H4 mid-CoT) use base models with explicit step-by-step prompts and Qwen3 instruct with thinking on as a separate, later cell. |
| Q6 | 2026-09-12 | Cross-lingual units: which languages? | es, de, fr (unit words are near-cognates; enough to test lexeme-independence without needing multilingual instruction following). |
| Q7 | 2026-09-12 | Git remote / push cadence? | Commit locally at each milestone; push when the PI gives the remote. |
