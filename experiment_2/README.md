# Experiment 2 Continuous XNOR

This keeps the same XNOR rule but varies coordinate magnitudes continuously,
allowing evaluation on unseen inputs. This folder was previously continuous_xnor.

Run these notebooks in order, each from a fresh kernel:

1. **00** — Data, mathematical checks, and examples.
2. **01** — Feature-only baselines and live training graphs.
3. **02** — Ordinary scaling, inverse initialization, and compensated SGD.
4. **03** — Saved results, prediction analysis, and comparison graphs.

Results stay in `results`; graphs are under `results/figures`. Full settings
use five seeds and 500 updates. Set QUICK_RUN in 01/02 and READ_PREVIEW in 03
to True for a separate preview under `results/preview`.

Use the shared root `.venv`. Start Jupyter from SimpBias or experiment_2.
Restart kernels opened before the rename. Training replaces existing results.
