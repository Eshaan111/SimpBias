# When Does Scaling Overcome Simplicity Bias

I built this project to test whether a neural network's preference for a simple predictive feature changes when I increase the numerical scale of an equally informative nonlinear feature. I then use matched initialization and compensated SGD to investigate why that preference changes.

The repository contains the experiment code, explanatory notebooks, completed results, and my written analysis. It is a small controlled study of feature competition, rather than a benchmark for realistic generalization.

## Start here

- **Read the study:** `When Does Scaling Overcome Simplicity Bias Analysis.docx` explains the question, mathematical reasoning, controls, graphs, and conclusions.
- **Inspect the final results:** `results/final_summary.csv` contains the 90 joint-model runs. `results/all_runs.csv` contains their 1,890 training checkpoints.
- **View the graphs without retraining:** run Notebook 08, followed by Notebook 09.
- **Understand the construction:** read Notebooks 00–06 in order. Notebook 02 contains the matched-budget feature-only baselines.
- **Reproduce the main sweep:** run `python run_all_experiments.py` from this folder.
- **Check the apparatus and saved results:** run `python verify_experiments.py`.

The controls are distinct experimental conditions. Ordinary scaling is the untreated condition; inverse initialization matches the starting function; compensated SGD intentionally combines inverse initialization with a corrected complex-column update. Standardization is a separate input check and is not applied to the main sweep.

## Environment and installation

Run commands from this project folder so imports from `src` and relative result paths resolve correctly. The data are generated locally; no external dataset, model download, service, or API key is needed.

A project-local environment can be created on Windows with:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Choose that environment as the notebook kernel. If using an existing environment, install the requirements there and launch Jupyter from this folder. The packages are PyTorch, NumPy, pandas, Matplotlib, nbformat, and JupyterLab. CPU is sufficient.

The final verification used Python 3.14.2, PyTorch 2.14.1+cpu, NumPy 2.5.4, pandas 3.0.6, and Matplotlib 3.11.2. These are the observed local versions, not a pinned environment lockfile. Results can differ slightly with other library versions or numerical settings.

## The question and hypotheses

My question is: holding feature-label information and nonlinear structure fixed, can positive scaling of a complex feature change a small ReLU MLP's diagnostic preference, and can SGD parameterization account for that change?

- **Hypothesis:** increasing complex-feature scale can move conflict preference toward the complex rule under a fixed global SGD learning rate.
- **Null hypothesis:** there is no systematic scale-dependent change under this setup across paired initialization seeds.
- **Alternative explanation:** the apparent preference may depend on initialization and effective optimization steps, rather than linear separability alone.

The current analysis is descriptive. The scale grid and five seeds provide a pilot; I do not report a formal hypothesis test, confidence interval, or a preregistered result.

## Data and model

For each example I sample `Y` uniformly from `{0, 1}` and independently sample `U` uniformly from `{-1, +1}`. Then:

```text
T = 2Y - 1
Xs = T
Xc = (U, UT)
X_alpha = [T, alpha*U, alpha*U*T], with alpha > 0
```

The population is balanced. The finite dataset is sampled randomly and is not forced to have equal counts of every pattern.

| Y | U | Unscaled input | Complex product |
|---|---|---|---|
| 0 | -1 | [-1, -1, +1] | -1 |
| 0 | +1 | [-1, +1, -1] | -1 |
| 1 | -1 | [+1, -1, -1] | +1 |
| 1 | +1 | [+1, +1, +1] | +1 |

The simple rule checks whether `Xs > 0`. The complex rule checks whether the two complex signs agree: same signs give class 1, opposite signs give class 0. This is **XNOR**, which is not linearly separable on the four corners. Multiplying both complex coordinates by positive alpha preserves the rule because their product becomes `alpha^2*T`.

Both feature groups recover the target perfectly. The model receives the coordinates, not their product. There are only four distinct ordinary input patterns at any fixed scale, so 1,024 samples repeat those patterns.

The joint model is `Linear(3, 32) -> ReLU -> Linear(32, 1)`. With samples stored as rows:

```text
Z1 = X_alpha @ W1.T + b1
H = ReLU(Z1)
logits = H @ W2.T + b2
```

`BCEWithLogitsLoss` receives logits directly and includes the sigmoid calculation in the loss. Evaluation uses `sigmoid(logits)` for probabilities and `logits >= 0` for predicted class 1.

## Training configuration

| Setting | Main sweep |
|---|---|
| Samples | 1,024; fixed data seed 11 |
| Initialization seeds | 42, 43, 44, 45, 46 |
| Complex scales | 0.1, 0.5, 1, 2, 5, 10 |
| Hidden width | 32 |
| Optimizer | Full-batch plain SGD |
| Global learning rate | 0.03 |
| Updates | 500 |
| Recorded checkpoints | Step 0, then every 25 updates through step 500 |
| Momentum, weight decay, clipping | None |

I reuse the same underlying examples and pair model seeds across conditions and scales. The simple-only and complex-only baselines use the same budget at alpha = 1, with one or two input coordinates respectively. They test separate learnability, not matched parameter counts or identical joint optimization.

## Why the controls work

The first-layer weight matrix has one simple column and two complex columns. I write the complex slice as `Wc`.

**Ordinary scaling** changes the complex input from `Xc` to `alpha*Xc`, keeping the same numerical initialization. It can change initial logits, ReLU gates, loss, and gradients. This measures the total scaling effect.

**Inverse initialization** copies a reference model and divides only its first-layer complex columns by alpha:

```text
Wc_scaled = Wc_reference / alpha
(alpha*Xc) @ Wc_scaled.T = Xc @ Wc_reference.T
```

The weighted contribution cancels, so initial preactivations, gates, logits, and loss match. All other parameters remain identical. This removes differences in the starting function, but does not match ordinary SGD updates.

**Compensated SGD** corrects those updates. Define the effective complex weights on the unscaled inputs as `Vc = alpha*Wc_scaled`. At a matched state, let `g` be the reference gradient on the complex columns:

```text
gradient on Wc_scaled = alpha*g
ordinary SGD: delta_Wc_scaled = -eta*alpha*g
effective change: delta_Vc = alpha*delta_Wc_scaled = -eta*alpha^2*g
```

One alpha comes from the scaled input in backpropagation. The second comes from converting the stored weight change into an effective weight change. I therefore use `eta/alpha^2` for the complex columns, implemented by dividing their gradients by `alpha^2` **after backpropagation and before the optimizer step**. Biases, the simple column, and output weights retain the original learning rate.

Combined with inverse initialization, this preserves the effective SGD trajectory in exact arithmetic for this setup. It is a mathematical sanity control. The alpha-squared identity applies at matched effective parameters; it is not a general claim about ordinary scaling's gradients or a claim that loss decreases alpha-squared times faster.

**Standardization** checks that `(alpha*x - alpha*mean)/(alpha*std)` equals the unscaled standardized coordinate. Notebook 06 includes a small epsilon, so agreement is approximate. This preprocessing would remove the intended input-scale intervention.

## What the evaluation metrics mean

**Training accuracy and BCE** are measured on the same sampled data used for optimization. They are not held-out generalization results. Both rules agree there, so high accuracy cannot identify preference.

**Complex conflict agreement** (`complex_reliance` in the CSVs) uses four equally weighted contradictory probes:

```text
[t_simple, alpha*u, alpha*u*(-t_simple)]
```

The simple and complex rules imply opposite classes. The score is the fraction of model predictions following the complex rule: 0 favors the simple rule on every probe; 1 favors the complex rule on every probe. An individual score changes in increments of 0.25. A score of 0.5 does not establish equal internal use of both features.

**Counterfactual sensitivity** starts from each ordinary sample and changes one group at a time:

- Simple intervention: negate coordinate 0.
- Complex intervention: negate coordinate 2. Negating one complex coordinate reverses the XNOR label; negating both would preserve it.
- `simple_flip_rate` / `complex_flip_rate`: fraction whose predicted class changes.
- `simple_probability_change` / `complex_probability_change`: mean absolute change in the sigmoid probability of class 1.

For example, probability 0.9 becoming 0.6 gives a change of 0.3 and no hard flip; becoming 0.4 gives a change of 0.5 and a hard flip. A zero class-flip rate can therefore coexist with nonzero probability sensitivity.

Conflict probes and interventions are outside the joint training support. The hard flip tests reuse the same four contradictory prototypes, so their agreement with conflict preference is not independent confirmation. Counterfactual means weight sampled examples, whereas conflict agreement weights the four probes equally. Plot shading shows one sample standard deviation across seeds, not a confidence interval.

## Notebook guide

| Notebook | Role | Main output or check |
|---|---|---|
| 00 Data Creation | Data and label rules | Four patterns and scale invariance |
| 01 Model Definition | Forward pass and gradients | Tensor shapes, ReLU gates, first-layer gradients |
| 02 Feature Only Baselines | Five-seed matched-budget capability checks | `02_feature_only_summary.csv`, `02_feature_only_history.csv` |
| 03 Hypothesis Ordinary Scaling | Seed-42 demonstration across scales | `03_ordinary.csv` |
| 04 Control Inverse Initialization | Seed-42 matched-function and gradient checks | `04_inverse.csv`; alpha 1 versus 5 |
| 05 Control Compensated SGD | Seed-42 trajectory-metric and final-logit checks | `05_compensated.csv`; alpha 1 versus 5 |
| 06 Control Standardization | Separate preprocessing check | Standardized-input equivalence |
| 07 Compare Experiments | Compare the demonstrations from 03–05 | Printed single-seed comparison |
| 08 Visualize CSV Results | Analyze the complete saved sweep | Graphs and `aggregated_research_summary.csv` |
| 09 Counterfactual Sensitivity | Explain and inspect the prediction interventions | Four-panel flip/probability graph and per-seed table |

Each notebook can run from a fresh kernel. Notebooks 03–05 are explanatory demonstrations, not the source of the five-seed thesis results. Notebooks 07–09 read saved CSVs and do not train models. Notebook 09 saves `results/figures/combined_counterfactual_sensitivity.png`.

## Reproduce or inspect the results

To inspect existing results, open Notebooks 08 and 09. There is no need to retrain.

To regenerate the current research results:

1. Run Notebook 02 to regenerate the feature-only baselines.
2. Run the main runner to regenerate the three-condition sweep:

   ```powershell
   python run_all_experiments.py
   ```

3. Run Notebooks 08 and 09 to regenerate summaries and graphs.
4. Run `python verify_experiments.py` to check the mathematics and the default saved-result grid.

The default runner **overwrites** `results/all_runs.csv` and `results/final_summary.csv`. For an additional exploratory run, use a separate output folder:

```powershell
python run_all_experiments.py --seeds 42 43 --alphas 0.5 1 5 --steps 100 --outdir results/exploratory
```

The visualization notebooks default to the main `results/` folder. Point their input paths to the exploratory folder and choose an available plotting scale when analyzing a different grid. `verify_experiments.py` checks the published default grid and budgets; it is not a validator for arbitrary exploratory configurations.

## Current findings

| Alpha | Ordinary conflict agreement | Inverse initialization | Compensated SGD |
|---|---|---|---|
| 0.1 | 0.00 | 0.00 | 0.00 |
| 0.5 | 0.00 | 0.00 | 0.00 |
| 1 | 0.00 | 0.00 | 0.00 |
| 2 | 0.85 | 0.05 | 0.00 |
| 5 | 1.00 | 0.95 | 0.00 |
| 10 | 1.00 | 1.00 | 0.00 |

These are means across five initialization seeds at step 500. All 90 joint runs and all 10 feature-only baselines reach 100% training accuracy.

My conclusions are:

- Positive scale changes diagnostic preference without changing the label information or XNOR structure.
- Different initial predictions are not necessary for reversal, although initialization affects where the transition occurs.
- Preference can continue changing after accuracy becomes perfect. At alpha = 5, all inverse models are correct by logged step 25, while mean conflict agreement subsequently rises from 0.45 to 0.95.
- Hard decisions can hide residual sensitivity. At alpha = 5, the mean simple probability change is about 0.167 under inverse initialization versus 0.008 under ordinary scaling.
- Compensation restores reference behavior in both hard and soft prediction metrics. This supports the effective-update explanation under the tested assumptions, while its equivalence is deliberately constructed.

The study does not establish unique on-distribution causal reliance, prove equal optimization difficulty, refute the cited theorems, or generalize to other networks, optimizers, and realistic distributions. The complex feature also has two coordinates while the simple feature has one.

## Which files matter

| Files | Purpose |
|---|---|
| `src/experiment.py`, `src/__init__.py` | Shared experiment implementation |
| `run_all_experiments.py` | Complete trained sweep |
| `verify_experiments.py` | Read-only mathematical and default saved-result checks |
| `results/final_summary.csv`, `results/all_runs.csv` | Current primary results and training histories |
| `results/02_feature_only_summary.csv`, `results/02_feature_only_history.csv` | Matched-budget baseline results and histories |
| `results/aggregated_research_summary.csv` | Derived means and standard deviations |
| `results/figures/` | Derived graphs, including counterfactual plots |
| `results/03_ordinary.csv`, `04_inverse.csv`, `05_compensated.csv` | Single-seed demonstration outputs |
| `baseline.py` | Optional random-initialization diagnostics and manual one-step control checks; separate from Notebook 02 |
| `.document_qa/`, `.validation/` | Document/rendering and verification intermediates; not primary research data |

The CSVs store prediction summaries, not every sample's logits or probabilities. Trained weight checkpoints are not saved. Raw data are regenerated from the documented construction and seed.

For the optional pre-training diagnostics:

```powershell
python baseline.py --output results/pretrain
```

This writes untrained metrics, one-step control errors, and metadata in a separate folder. Its sample-weighted conflict construction and complex-coordinate flip convention differ from the main uniform four-probe diagnostic; I do not combine these outputs with the trained sweep.

## Final verification

On 13 October 2026, I checked the project by executing all ten notebooks from fresh kernels in an isolated copy and rerunning the complete default sweep. Both primary CSVs reproduced exactly across all fields: 90 final rows and 1,890 history rows. The ten feature-only baseline runs also reproduced exactly.

The mathematical checks cover both label rules, scaling invariance, the ideal-rule intervention responses, standardization, the analytical BCE/ReLU first-layer gradient, inverse initialization, the alpha-squared effective update, and the actual compensated training implementation across all 30 seed/scale pairs. Saved-result checks cover grid completeness, budgets, metric ranges, final/history agreement, aggregates, and compensated reference histories.

Verification intermediates are in `.validation/`. The primary research CSVs were preserved; notebook outputs and derived figures were refreshed. To repeat the read-only mathematical and saved-result checks, run `python verify_experiments.py`.

## References

- Shah et al. (2020), [The Pitfalls of Simplicity Bias in Neural Networks](https://papers.nips.cc/paper/2020/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html).
- Morwani et al. (2023), [Simplicity Bias in 1-Hidden Layer Neural Networks](https://papers.neurips.cc/paper_files/paper/2023/hash/196c4e02b7464c554f0f5646af5d502e-Abstract-Conference.html).
- Rahaman et al. (2019), [On the Spectral Bias of Neural Networks](https://proceedings.mlr.press/v97/rahaman19a.html).
# SimpBias
