# uot / t1 — Do language models have an internal algebra of physical units?

Attempt t1 (2026-09-12 → 13). Start with `reports/TECHNICAL_REPORT.md`; the audit trail is `NOTEBOOK.md` (dated entries, FLAGS at the top), `CLAIMS.md` (every claim → status → experiment IDs), `PREDICTIONS_phase{0,1,2,3}.md` (pre-registered, never edited after commit), `decisions/DR-00*.md`, and the gate reports + red-team appendices under `reports/`. `NEXT.md` lists what to do with more time.

## Layout
| path | what |
|---|---|
| `src/uot/` | library: `dims` (exponent-vector algebra), `units` (Pint-validated registry, invented lexemes, compound rendering), `quantity`, `lexicon`, `tasks/` (T1–T5 generators, Phase-1 mentions, Phase-2 interchange), `parse_units` (free-form unit expression → dimension), `lm/` (loading, candidate scoring, generation), `cache` (residual caching), `probes`, `das` (subspace interchange), `frequency` (infini-gram) |
| `scripts/phase0..3/` | numbered entry points per experiment ID (see NOTEBOOK) |
| `data/` | stimulus sets (`phase0/`, `phase1/`, `phase2/`, `phase3/`), Dolma-1.7 frequency table, corpus-verified lexeme pool, wordlist |
| `runs/` | per-item results, probe/DAS/control JSON, analysis CSVs (large arrays git-ignored) |
| `cache/` | residual-stream caches (git-ignored; regenerate with `scripts/phase1/02_cache_residuals.py`) |
| `figures/phase0..3/` | all figures; `make figures` regenerates them from `runs/` |
| `env_setup/` | conda recipe, frozen requirements, driver/torch capture, model download script |
| `papers/` | reading notes live in NOTEBOOK; PDFs git-ignored |

## Reproduce
```bash
bash env_setup/create_env.sh            # conda env at t1/env (torch cu124 for driver 550)
env/bin/python -m pytest -q tests        # 23 tests
make figures                             # regenerates every figure/table from cached runs
# GPU stages (rosetta18 launchers): scripts/phase0/run_E0.3_gpu.sh, scripts/phase1/run_cache.sh,
# scripts/phase1/run_probes_par.sh, scripts/phase2/run_das_sweep.sh, scripts/phase2/run_seeds_controls.sh
```
Models: Qwen3-4B/8B/14B-Base, OLMo-3-7B/32B, Gemma-2-9B, Qwen3-4B/8B instruct (HF ids in `src/uot/lm/models.py`).
