# Regenerate every figure and table from cached runs (no GPU needed).
PY=env/bin/python
.PHONY: figures tests
figures:
	$(PY) scripts/phase0/05_analyze_battery.py --runs runs/E0.3
	$(PY) scripts/phase0/05_analyze_battery.py --runs runs/E0.3i
	$(PY) scripts/phase0/05_analyze_battery.py --runs runs/E1.0b
	$(PY) scripts/phase0/07_figures.py
	$(PY) scripts/phase1/05_probe_figures.py --csv runs/E1.1/qwen3-4b-base/probes_P1_s0_main_s0_*.csv
	$(PY) scripts/phase1/11_g1_figures.py
	$(PY) scripts/phase2/04_summarize_das.py
	$(PY) scripts/phase2/07_figures.py
tests:
	$(PY) -m pytest -q tests
