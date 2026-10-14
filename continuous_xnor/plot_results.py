"""Read the new CSVs and plot them. This file never trains a model."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {'ordinary': 'Ordinary scaling', 'inverse_init': 'Inverse initialization',
          'compensated_sgd': 'Compensated SGD'}
COLORS = {'ordinary': 'tab:blue', 'inverse_init': 'tab:orange', 'compensated_sgd': 'tab:green'}


def plot_scales(ax, final, metric, title, bounded=True):
    for condition, label in LABELS.items():
        group = final[final.condition == condition].groupby('alpha')[metric].agg(['mean', 'std'])
        x = group.index.to_numpy()
        mean = group['mean'].to_numpy()
        sd = group['std'].fillna(0).to_numpy()
        lower, upper = mean - sd, mean + sd
        if bounded:
            lower, upper = lower.clip(0, 1), upper.clip(0, 1)
        ax.plot(x, mean, 'o-', label=label, color=COLORS[condition])
        ax.fill_between(x, lower, upper, alpha=0.15, color=COLORS[condition])
    scales = sorted(final.alpha.unique())
    ax.set_xscale('log')
    ax.set_xticks(scales, labels=[f'{a:g}' for a in scales])
    ax.set(title=title, xlabel='Complex feature scale alpha')
    if bounded:
        ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    final = pd.read_csv(args.results / 'final_summary.csv')
    figures = args.results / 'figures'
    figures.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title, bounded in [
        (axes[0], 'complex_conflict_agreement', 'Complex conflict agreement', True),
        (axes[1], 'test_accuracy', 'Held-out accuracy', True),
        (axes[2], 'test_loss', 'Held-out BCE loss', False),
    ]:
        plot_scales(ax, final, metric, title, bounded)
    fig.tight_layout()
    fig.savefig(figures / 'final_results.png', dpi=170)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ('test_simple_flip_rate', 'Simple hard-flip rate'),
        ('test_complex_flip_rate', 'Complex hard-flip rate'),
        ('test_simple_probability_change', 'Simple mean absolute probability change'),
        ('test_complex_probability_change', 'Complex mean absolute probability change'),
    ]
    for ax, (metric, title) in zip(axes.flat, metrics):
        plot_scales(ax, final, metric, title)
    fig.tight_layout()
    fig.savefig(figures / 'test_counterfactuals.png', dpi=170)
    print('Saved figures to', figures.resolve())
    print('Lines are seed means. Shading is one sample standard deviation, not a confidence interval.')
    plt.show()


if __name__ == '__main__':
    main()
