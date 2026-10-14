# Continuous XNOR notebooks

Run the numbered notebooks top to bottom. Each can start from a fresh kernel;
no notebook depends on variables left in another notebook.

| Notebook | What you see |
|---|---|
| `00_Data_and_Mathematical_Checks.ipynb` | Continuous samples, label rules, data scatter plots, conflict examples, and initial-function/gradient checks |
| `01_Feature_Only_Baselines.ipynb` | Simple-only and complex-only models, live training curves, held-out results, and baseline comparisons |
| `02_Scaling_and_Controls.ipynb` | The ordinary, inverse-initialized, and compensated conditions, live progress, and reference-equivalence checks |
| `03_Results_and_Prediction_Analysis.ipynb` | Final scale comparisons, flip/probability graphs, learning curves, seed-level tables, and exported figures |

## Open the notebooks

From this folder, using a Python environment with PyTorch:

```powershell
python -m pip install -r requirements.txt
python -m jupyter lab
```

Select that environment as the kernel. The notebook version is self-contained:
`notebook_experiment.py` supplies its data, model, and training functions, and
the notebooks display important function definitions beside explanations.
It does not import the discrete experiment or require the earlier runner scripts.

## Watch training

In Notebooks 01 and 02:

- Keep `QUICK_RUN = False` for the research run: five seeds and 500 updates.
- Set `SHOW_LIVE = True` to watch loss and accuracy. Notebook 02 also shows
  conflict preference for the current model, with run number, scale, and seed.
- The live display redraws every 100 updates and at the final update. All
  checkpoint metrics are still recorded every 25 updates.
- Set `SHOW_LIVE = False` to skip live redraws and print each completed run.
  This changes the display only, not training or evaluation.

The display follows one model at a time. The later summary cells and Notebook 03
compare all seeds and conditions, so earlier runs are still available after the
live display advances.

## Preview without replacing research results

Set `QUICK_RUN = True` in both training notebooks for a one-seed, 25-update preview.
Notebook 02 then uses scales 1 and 5. Preview output is saved to `results/preview/`.
In Notebook 03, set `READ_PREVIEW = True` to read those files.

For the full study, use `QUICK_RUN = False` in Notebooks 01 and 02 and
`READ_PREVIEW = False` in Notebook 03. Keep steps, learning rate, and seeds matched
between the baseline and main experiment. Notebook 03 checks those settings.
A preview checks execution; it does not establish learnability or bias.

## What changes from the discrete experiment

The input is `[T*As, alpha*U*A1, alpha*U*T*A2]`, where `T = 2Y - 1`, the nuisance
sign `U` is independent of Y, and all magnitudes are independently sampled from
Uniform(0.5, 1.5). The population is balanced; finite training samples are not
forced to be exactly balanced. Both feature groups still recover every label.

There is no normalization. The same base signs and magnitudes are reused across
every scale, seed, and condition. Complex inputs are multiplied by alpha only
after those samples are generated.

Independent train, validation, and test splits use seeds 11, 12, and 13, with
1,024, 1,024, and 2,048 samples. A separate seed 14 supplies 2,048 conflict probes,
balanced across the four sign configurations. The probes vary in magnitude and
make the simple and complex rules disagree.

The model remains Linear(3, 32), ReLU, Linear(32, 1), with BCE-with-logits loss and
full-batch SGD at learning rate 0.03. The default scales are 0.1, 0.5, 1, 2, 5, 10.
There is no momentum, weight decay, clipping, or early stopping. The two feature-only
baselines use the same width and budget at alpha 1, with their respective input sizes.

Inverse initialization divides only the first-layer complex weight columns by
alpha. Compensation additionally divides their gradients by alpha squared after
backpropagation and before the SGD step. Each seed has an auxiliary unscaled
reference run for checking final logits; it is not another row in the main sweep.

## Measurements and outputs

Training and validation loss/accuracy are recorded during optimization. Only
training labels enter backpropagation. The test split is evaluated after each
model completes its fixed update budget.

Conflict agreement measures how often a prediction follows the complex rule on
contradictory probes. Counterfactuals negate the simple coordinate or one complex
coordinate. Their metrics are the class-flip rate and mean absolute sigmoid
probability change. Validation sensitivities appear in the training history;
final `test_*` sensitivities use the independent test sample.

These interventions remain outside the joint training support. Their hard scores
are related behavioral probes, not independent proof of causal reliance. Plot
shading is one sample standard deviation across seeds, not a confidence interval.

The full notebooks save inside their own `results/` folder:

- `baseline_summary.csv`, `baseline_history.csv`, `baseline_settings.json`.
- `final_summary.csv`: 90 default joint runs.
- `training_history.csv`: 1,890 default checkpoints.
- `metadata.json`: experiment settings and dataset information.
- `train_data.csv`, `validation_data.csv`, `test_data.csv`, `conflicts_data.csv`:
  unscaled coordinates, with both implied labels stored for conflicts.
- `aggregated_summary.csv` and `figures/`: derived comparisons and graphs.

The main notebook checkpoints completed runs to CSV. Its metadata is finalized
only when the sweep completes. Finish Notebook 02 before interpreting its saved
grid. Rerunning the full notebooks overwrites this new experiment's own outputs.

Held-out coordinates address the four-prototype limitation within the chosen
distribution. They do not test extrapolation outside the magnitude range,
equalize group dimensions, or compare different nonlinear rules. Inspect the
complex-only baseline and mathematical controls before interpreting preference.
