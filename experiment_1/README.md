# Experiment 1 Discrete XNOR

This tests how scaling changes preference between a simple sign and an XNOR
pair. The data repeat four distinct input patterns.

- Read notebooks 00–06 for the data, model, baselines, and controls.
- Run 02 for feature-only baselines; 03–05 are single-seed demonstrations.
- Run 08 and 09 to view the complete saved study and prediction graphs.
- From SimpBias, run `.venv\Scripts\python.exe experiment_1\run_all_experiments.py`
  to regenerate the five-seed study. This replaces the main result CSVs.
- Check it with `.venv\Scripts\python.exe experiment_1\verify_experiments.py`.

Code is in `src`; all results and graphs stay in this folder's `results`.
Use the shared root `.venv`. Start notebooks from SimpBias or experiment_1,
with a fresh kernel. Restart any kernel opened before the folders were moved.
