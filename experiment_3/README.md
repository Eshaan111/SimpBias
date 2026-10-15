# Experiment 3 Extra coordinate and equal supplied dimensions

This compares continuous XNOR with no extra coordinate (C0), a zero coordinate
(C1), or independent noise (C2) added to the simple group. Informative dimensions
remain unequal; C1 is a sanity check and C1 versus C2 tests nuisance variation.

Run these notebooks in order, each from a fresh kernel:

1. **00** — Data, label rules, and mathematical checks.
2. **01** — Feature-only baselines and live training graphs.
3. **02** — The 270-model scaling/control study.
4. **03** — Results, paired noise comparisons, and graphs.

Results stay in `results`; graphs are under `results/figures`. Set QUICK_RUN
in 01/02 and READ_PREVIEW in 03 to True for a separate short preview.
Use the shared root `.venv` and start Jupyter from SimpBias or experiment_3.
Restart kernels opened before the folder move. Training replaces existing results.
