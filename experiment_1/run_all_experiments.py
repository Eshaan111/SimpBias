"""Run the simplicity-bias experiments across seeds and feature scales.

From the experiment_1 folder:
    python run_all_experiments.py
    python run_all_experiments.py --seeds 42 43 --alphas 0.5 1 5 --steps 100

Writes results/all_runs.csv (training curves) and results/final_summary.csv.
Uses the same full-batch SGD implementation as the notebooks.
"""
import argparse
import csv
from pathlib import Path

import torch
from src.experiment import (
    make_data, make_model, inverse_initialize, train,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def initialization_checks(ref, scaled, x_ref, x_scaled, y, alpha):
    """Check inverse initialization BEFORE training changes the parameters."""
    with torch.no_grad():
        z_error = (ref.fc1(x_ref) - scaled.fc1(x_scaled)).abs().max().item()
        logit_error = (ref(x_ref) - scaled(x_scaled)).abs().max().item()

    ref.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn(ref(x_ref), y).backward()
    loss_fn(scaled(x_scaled), y).backward()
    grad_error = (
        scaled.fc1.weight.grad[:, 1:3] - alpha * ref.fc1.weight.grad[:, 1:3]
    ).abs().max().item()
    ref.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    return z_error, logit_error, grad_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 5, 10])
    parser.add_argument('--data-seed', type=int, default=11)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.03)
    parser.add_argument('--record-every', type=int, default=25)
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()

    if any(alpha <= 0 for alpha in args.alphas):
        parser.error('All alpha values must be positive.')
    if args.steps < 0 or args.record_every <= 0 or args.n <= 0 or args.lr <= 0:
        parser.error('steps must be >= 0; record-every, n, and lr must be > 0.')

    # Generate one common dataset per scale, so only alpha changes.
    x_ref, y = make_data(args.n, seed=args.data_seed, alpha=1.0)
    data = {a: make_data(args.n, seed=args.data_seed, alpha=a)[0]
            for a in args.alphas}

    histories = []
    finals = []
    experiments = ('ordinary', 'inverse_init', 'compensated_sgd')
    total = len(args.seeds) * len(args.alphas) * len(experiments)
    count = 0

    for seed in args.seeds:
        reference_final = make_model(seed=seed)
        train(reference_final, x_ref, y, steps=args.steps, lr=args.lr,
              alpha=1.0, record_every=max(1, args.steps))
        with torch.no_grad():
            reference_final_logits = reference_final(x_ref)

        for alpha in args.alphas:
            x = data[alpha]
            for experiment in experiments:
                count += 1
                model = make_model(seed=seed)
                z_error = logit_error = grad_error = None

                if experiment != 'ordinary':
                    reference = make_model(seed=seed)
                    inverse_initialize(model, reference, alpha)
                    z_error, logit_error, grad_error = initialization_checks(
                        reference, model, x_ref, x, y, alpha
                    )

                history = train(
                    model, x, y, steps=args.steps, lr=args.lr,
                    alpha=alpha, compensate=(experiment == 'compensated_sgd'),
                    record_every=args.record_every,
                )

                common = {
                    'experiment': experiment,
                    'seed': seed,
                    'data_seed': args.data_seed,
                    'alpha': alpha,
                    'n': args.n,
                    'lr': args.lr,
                    'steps': args.steps,
                    'initial_preactivation_max_error': z_error,
                    'initial_logit_max_error': logit_error,
                    'initial_gradient_scaling_max_error': grad_error,
                }
                for point in history:
                    histories.append({**common, **point})

                # Compare final logits against the alpha=1 SGD reference.
                with torch.no_grad():
                    final_logit_error = (
                        model(x) - reference_final_logits
                    ).abs().max().item()

                finals.append({
                    **common,
                    **history[-1],
                    'final_logit_max_error_vs_alpha1_reference': final_logit_error,
                })
                print(
                    f'[{count:>3}/{total}] {experiment:16s} '
                    f'seed={seed:<4} alpha={alpha:<5g} '
                    f'loss={history[-1]["loss"]:.4f} '
                    f'acc={history[-1]["accuracy"]:.3f} '
                    f'complex_reliance={history[-1]["complex_reliance"]:.2f}',
                    flush=True,
                )

    write_csv(args.outdir / 'all_runs.csv', histories)
    write_csv(args.outdir / 'final_summary.csv', finals)
    print(f'\nSaved {len(histories)} training-history rows to {args.outdir / "all_runs.csv"}')
    print(f'Saved {len(finals)} final rows to {args.outdir / "final_summary.csv"}')
    print('Note: conflict reliance uses four out-of-distribution examples.')


if __name__ == '__main__':
    main()
