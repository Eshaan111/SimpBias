# Experiment 3 Equal dimensional continuous XNOR

I test whether adding an irrelevant coordinate to the simple group changes the
scaling effect found in continuous XNOR. This folder is independent of earlier
experiments. Their code and results are not modified.

| Input condition | Input | Purpose |
|---|---|---|
| C0 original | `[AT, alpha*BU, alpha*CUT]` | Existing continuous design |
| C1 zero | `[AT, 0, alpha*BU, alpha*CUT]` | Inactive-coordinate sanity check |
| C2 noise | `[AT, R, alpha*BU, alpha*CUT]` | Active irrelevant variation |

The same underlying samples and labels are used across conditions and scales.
Magnitudes A/B/C are independently Uniform(0.5,1.5); R is independent Gaussian
noise. There is no normalization. Both predictive rules remain unchanged.

## Run these notebooks in order

| Notebook | Contents |
|---|---|
| `00_Data_and_Mathematical_Checks.ipynb` | Data examples, scatter plots, rules, initial matching, compensation checks |
| `01_Feature_Only_Baselines.ipynb` | Separate learnability, live training graphs, held-out performance |
| `02_Scaling_and_Controls.ipynb` | Visible experiment loops, live training, saved measurements |
| `03_Results_and_Prediction_Analysis.ipynb` | Comparison graphs, paired noise effects, prediction analysis |

Each notebook works from a fresh kernel. `experiment.py` holds short shared
functions for data, models, training, and saving files. Important functions are
displayed inside the notebooks beside their mathematical explanations.

From this folder, using a Python environment with PyTorch:

```powershell
python -m pip install -r requirements.txt
python -m jupyter lab
```

Select that environment as the kernel. Start Jupyter in this folder or its
project parent and run the notebooks top to bottom.

## Settings

- Five model seeds: 42–46. Scales: 0.1, 0.5, 1, 2, 5, 10.
- Width 32 ReLU hidden layer, one output logit, BCEWithLogitsLoss.
- Plain full-batch SGD, learning rate 0.03, 500 updates.
- No momentum, weight decay, clipping, normalization, or early stopping.
- Train/validation/test: 1024/1024/2048 samples, seeds 11/12/13.
- Conflict probes: 2048 samples, seed 14, 512 per sign configuration.
- Checkpoints every 25 updates, including step zero.
- 270 primary models, 15 unit-scale trajectory references, and 20 baselines.
  The identical complex-only baseline is trained once per seed.

Set `SHOW_LIVE = True` to watch training. It redraws every 100 updates and at
completion; all 25-update checkpoints are still saved.

For a short preview, set `QUICK_RUN = True` in 01 and 02 and `READ_PREVIEW = True`
in 03. This uses one seed, 25 updates, and scales 1/5. Preview output goes to
`results/preview`; full-study output goes to `results`. Keep all three flags
False for the complete study. Preview results are not research conclusions.

## Controls to keep intact

- Shared weights, biases, and output parameters are explicitly copied across C0/C1/C2.
- The nuisance column is identical in C1/C2. Its initialization uses Uniform
  bounds ±1/sqrt(3), matching the three-input reference columns. Noise samples
  and nuisance weights use independent random generators.
- Complex columns are `1:3` for C0 and `2:4` for C1/C2.
- Inverse initialization divides only complex weights by alpha.
- Compensated SGD also divides only complex gradients by alpha squared.
- C0/C1 must match. Each compensated model must match its own condition's
  alpha=1 reference, including the nuisance column where present.
- Initial tolerance: 1e-6. Trajectory tolerance: 1e-4 maximum absolute error,
  checked on effective parameters and validation/conflict logits at checkpoints.

## Results and graphs

The notebooks save final results in `summary.csv`, checkpoint measurements in
`all_runs.csv`, numerical controls in `trajectory_checks.csv`, and baselines in
`feature_only.csv` and `feature_only_history.csv`. Settings and completion state
are stored in JSON. Unscaled split CSVs include the shared noise values.

Notebook 03 displays and saves graphs in `results/figures` (or preview/figures):

- Conflict preference versus scale for C0/C1/C2.
- Held-out accuracy and BCE versus scale.
- Simple and complex hard flip rates and probability changes.
- Validation accuracy and conflict preference during training.
- Numerical trajectory discrepancies.

Results are saved after each model. An error stops execution; the completion
marker is set only after the whole notebook succeeds. Analysis checks the
expected model keys and checkpoints and rejects incomplete studies. Re-running
replaces results in the selected folder; it does not resume. If the dataset or
helper code changes, rerun both training notebooks before analysis.

C0/C1 is a sanity check. C1/C2 is the nuisance comparison. Equal supplied
dimensions do not equalize informative dimensions. Conflict agreement is a
diagnostic preference score on contradictory inputs, separate from IID accuracy.
Simple agreement is one minus complex agreement. Counterfactuals keep noise and
magnitudes fixed. Bands show sample standard deviation, not confidence intervals.
One fixed dataset draw and five model seeds do not quantify dataset uncertainty.
