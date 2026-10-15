"""Run this file to generate continuous-XNOR results in this folder only."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from experiment import (
    make_data, make_conflicts, scale_complex, make_model, inverse_initialize,
    train_model, classification_metrics, counterfactual_metrics, initial_checks,
)

# Keep the original study's architecture and optimizer budget.
N_TRAIN = 1024
N_VALIDATION = 1024
N_TEST = 2048
N_CONFLICTS = 2048
DATA_SEEDS = {'train': 11, 'validation': 12, 'test': 13, 'conflicts': 14}
HERE = Path(__file__).resolve().parent


def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_data(path, data, conflict=False):
    x, y = data
    # Save unscaled coordinates so the identical samples can be inspected.
    label_name = 'complex_implied_label' if conflict else 'label'
    rows = []
    for coordinates, label in zip(x.tolist(), y[:, 0].tolist()):
        row = dict(zip(['simple', 'complex_1', 'complex_2'], coordinates))
        row[label_name] = int(label)
        if conflict:
            row['simple_implied_label'] = 1 - int(label)
        rows.append(row)
    save_csv(path, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 5, 10])
    parser.add_argument('--outdir', type=Path, default=HERE / 'results')
    args = parser.parse_args()
    if args.steps < 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('steps must be nonnegative and lr must be finite and positive.')
    if any(not math.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error('All alpha values must be finite and positive.')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.alphas)) != len(args.alphas):
        parser.error('Do not repeat seeds or alpha values.')

    # CPU and one thread keep this small experiment predictable and lightweight.
    torch.set_num_threads(1)
    args.outdir.mkdir(parents=True, exist_ok=True)
    train_data = make_data(N_TRAIN, DATA_SEEDS['train'])
    validation_data = make_data(N_VALIDATION, DATA_SEEDS['validation'])
    test_data = make_data(N_TEST, DATA_SEEDS['test'])
    conflicts = make_conflicts(N_CONFLICTS, DATA_SEEDS['conflicts'])
    for name, data in [('train', train_data), ('validation', validation_data), ('test', test_data)]:
        save_data(args.outdir / f'{name}_data.csv', data)
    save_data(args.outdir / 'conflict_probes.csv', conflicts, conflict=True)

    # Generate the data ONCE. Scaling never resamples signs or magnitudes.
    scaled_data = {}
    for alpha in args.alphas:
        scaled_data[alpha] = [
            (scale_complex(x, alpha), y)
            for x, y in [train_data, validation_data, test_data, conflicts]
        ]

    baseline_final, baseline_history = [], []
    final_rows, history_rows = [], []
    conditions = ['ordinary', 'inverse_init', 'compensated_sgd']

    for seed in args.seeds:
        # Learnability baseline: same width, learning rate, and update budget.
        for name, columns in [('simple_only', [0]), ('complex_only', [1, 2])]:
            model = make_model(seed, inputs=len(columns))
            train_subset = (train_data[0][:, columns], train_data[1])
            validation_subset = (validation_data[0][:, columns], validation_data[1])
            history = train_model(model, train_subset, validation_subset,
                                  steps=args.steps, lr=args.lr)
            info = {'condition': name, 'seed': seed, 'alpha': 1.0}
            baseline_history.extend({**info, **row} for row in history)
            final = {**info, **history[-1]}
            metrics = classification_metrics(model, test_data[0][:, columns], test_data[1])
            final.update({f'test_{key}': value for key, value in metrics.items()})
            baseline_final.append(final)
            print(f'{name}: seed={seed}, test accuracy={final["test_accuracy"]:.3f}', flush=True)

        initial_reference = make_model(seed)
        # An auxiliary alpha=1 run is needed even if --alphas excludes 1.
        final_reference = make_model(seed)
        train_model(final_reference, train_data, validation_data, steps=args.steps, lr=args.lr)
        with torch.no_grad():
            reference_test_logits = final_reference(test_data[0])

        for alpha in args.alphas:
            scaled_train, scaled_validation, scaled_test, scaled_conflicts = scaled_data[alpha]
            for condition in conditions:
                model = make_model(seed)
                checks = {
                    'initial_preactivation_max_error': None,
                    'initial_logit_max_error': None,
                    'initial_gradient_scaling_max_error': None,
                }
                if condition != 'ordinary':
                    inverse_initialize(model, initial_reference, alpha)
                    checks = initial_checks(initial_reference, model, train_data[0],
                                            scaled_train[0], train_data[1], alpha)
                history = train_model(
                    model, scaled_train, scaled_validation, steps=args.steps, lr=args.lr,
                    alpha=alpha, compensate=(condition == 'compensated_sgd'),
                    conflict_data=scaled_conflicts,
                )
                info = {'condition': condition, 'seed': seed, 'alpha': alpha}
                history_rows.extend({**info, **row} for row in history)
                final = {**info, **checks, **history[-1]}
                # Test labels are used only for this final evaluation, never SGD.
                metrics = classification_metrics(model, *scaled_test)
                metrics.update(counterfactual_metrics(model, scaled_test[0]))
                final.update({f'test_{key}': value for key, value in metrics.items()})
                with torch.no_grad():
                    error = (model(scaled_test[0]) - reference_test_logits).abs().max().item()
                final['final_test_logit_max_error_vs_alpha1'] = error
                final_rows.append(final)
                print(f'{condition}: seed={seed}, alpha={alpha:g}, '
                      f'test accuracy={final["test_accuracy"]:.3f}, '
                      f'complex agreement={final["complex_conflict_agreement"]:.3f}', flush=True)
                if condition == 'compensated_sgd' and error > 1e-4:
                    print(f'  Check compensation: reference logit error is {error:.3g}.', flush=True)

    save_csv(args.outdir / 'baseline_summary.csv', baseline_final)
    save_csv(args.outdir / 'baseline_history.csv', baseline_history)
    save_csv(args.outdir / 'final_summary.csv', final_rows)
    save_csv(args.outdir / 'training_history.csv', history_rows)
    metadata = {
        'dataset': 'continuous_xnor', 'magnitude_distribution': 'independent Uniform(0.5, 1.5)',
        'normalization': 'none', 'data_seeds': DATA_SEEDS,
        'samples': {'train': N_TRAIN, 'validation': N_VALIDATION, 'test': N_TEST,
                    'balanced_conflicts': N_CONFLICTS},
        'model': 'Linear(inputs, 32) -> ReLU -> Linear(32, 1)',
        'optimizer': 'full-batch SGD; no momentum, weight decay, or clipping',
        'lr': args.lr, 'steps': args.steps, 'record_every': 25,
        'seeds': args.seeds, 'alphas': args.alphas, 'conditions': conditions,
        'baseline_alpha': 1.0, 'auxiliary_alpha1_reference_runs': len(args.seeds),
        'torch_version': torch.__version__,
    }
    (args.outdir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('\nSaved results to', args.outdir.resolve())
    print('Inspect complex-only held-out accuracy before interpreting joint preference.')


if __name__ == '__main__':
    main()
