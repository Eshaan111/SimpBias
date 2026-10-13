"""Read-only mathematical and saved-result checks. Run from the project root."""
import ast
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.experiment import (
    accuracy, conflict_reliance, counterfactual_sensitivity,
    inverse_initialize, make_data, make_model, train,
)

ROOT = Path(__file__).resolve().parent
SCALES = [0.1, 0.5, 1, 2, 5, 10]
SEEDS = [42, 43, 44, 45, 46]
CONDITIONS = ['ordinary', 'inverse_init', 'compensated_sgd']


def mathematical_checks():
    """Check formulas independently of the optimizer's implementation."""
    x, y = make_data(1024, seed=11)
    assert torch.equal((x[:, :1] > 0).float(), y)
    assert torch.equal((x[:, 1:2] * x[:, 2:3] > 0).float(), y)
    assert len(torch.unique(x, dim=0)) == 4
    # Ideal feature rules must give opposite intervention responses.
    simple_rule = lambda z: z[:, :1]
    complex_rule = lambda z: z[:, 1:2] * z[:, 2:3]
    for rule, preference, sf, cf in [(simple_rule, 0., 1., 0.),
                                      (complex_rule, 1., 0., 1.)]:
        assert accuracy(rule(x), y) == 1
        assert conflict_reliance(rule) == preference
        metrics = counterfactual_sensitivity(rule, x)
        assert metrics['simple_flip_rate'] == sf
        assert metrics['complex_flip_rate'] == cf

    max_initial = max_gradient = max_trajectory = 0.
    for seed, alpha in itertools.product(SEEDS, SCALES):
        xa, ya = make_data(1024, seed=11, alpha=alpha)
        assert torch.equal(y, ya)
        torch.testing.assert_close(xa[:, :1], x[:, :1])
        torch.testing.assert_close(xa[:, 1:], alpha * x[:, 1:])
        standardize = lambda z: (z - z.mean(0)) / z.std(0, unbiased=False)
        torch.testing.assert_close(standardize(xa), standardize(x), atol=1e-6, rtol=1e-6)

        # Float64 makes the algebra check distinct from float32 rounding.
        ref = make_model(seed).double()
        scaled = inverse_initialize(make_model(seed).double(), ref, alpha)
        xd, xad, yd = x.double(), xa.double(), y.double()
        # alpha=0.1 was first rounded into float32 by the data constructor.
        xad[:, 1:] = xd[:, 1:] * alpha
        max_initial = max(max_initial, (ref(xd) - scaled(xad)).abs().max().item())
        loss = torch.nn.BCEWithLogitsLoss()
        loss(ref(xd), yd).backward()
        loss(scaled(xad), yd).backward()
        with torch.no_grad():
            z1 = ref.fc1(xd)
            upstream = ((torch.sigmoid(ref(xd)) - yd) / len(yd)) @ ref.fc2.weight
            g = (upstream * (z1 > 0)).T @ xd
            torch.testing.assert_close(ref.fc1.weight.grad, g, atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(scaled.fc1.weight.grad[:, 1:], alpha * g[:, 1:],
                                       atol=1e-12, rtol=1e-10)
            # One raw SGD update has alpha^2 times the effective complex update.
            actual_effective_delta = alpha * (-0.03 * scaled.fc1.weight.grad[:, 1:])
            expected_effective_delta = -0.03 * alpha**2 * g[:, 1:]
            torch.testing.assert_close(actual_effective_delta, expected_effective_delta,
                                       atol=1e-12, rtol=1e-10)
        max_gradient = max(max_gradient,
                           (scaled.fc1.weight.grad[:, 1:] - alpha * g[:, 1:]).abs().max().item())

        # Exercise the real shared train() implementation, checking every step.
        ref = make_model(seed)
        scaled = inverse_initialize(make_model(seed), ref, alpha)
        h1 = train(ref, x, y, steps=25, record_every=1)
        ha = train(scaled, xa, y, steps=25, alpha=alpha, compensate=True, record_every=1)
        for r1, ra in zip(h1, ha):
            for key in r1:
                assert np.isclose(r1[key], ra[key], atol=2e-6, rtol=1e-5), (seed, alpha, key)
        error = (ref(x) - scaled(xa)).abs().max().item()
        max_trajectory = max(max_trajectory, error)
        torch.testing.assert_close(ref(x), scaled(xa), atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(ref.fc1.weight[:, 1:], alpha * scaled.fc1.weight[:, 1:],
                                   atol=2e-6, rtol=1e-5)
        for name in ['fc1.bias', 'fc2.weight', 'fc2.bias']:
            torch.testing.assert_close(dict(ref.named_parameters())[name],
                                       dict(scaled.named_parameters())[name],
                                       atol=2e-6, rtol=1e-5)
    assert max_initial < 1e-12 and max_gradient < 1e-12
    print(f'PASS math: 30 seed/scale pairs; initial={max_initial:.3g}, '
          f'gradient={max_gradient:.3g}, compensated logits={max_trajectory:.3g}')


def saved_result_checks():
    def read(name):
        with (ROOT / 'results' / name).open(newline='', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    final, history = read('final_summary.csv'), read('all_runs.csv')
    key = lambda r: (r['experiment'], int(r['seed']), float(r['alpha']))
    expected = set(itertools.product(CONDITIONS, SEEDS, SCALES))
    assert len(final) == 90 and {key(r) for r in final} == expected
    assert len(history) == 1890
    points = {}
    bounded = ['accuracy', 'complex_reliance', 'simple_flip_rate', 'complex_flip_rate',
               'simple_probability_change', 'complex_probability_change']
    for row in history:
        k = key(row) + (int(row['step']),)
        assert k not in points, ('duplicate checkpoint', k)
        points[k] = row
        assert all(math.isfinite(float(row[c])) and 0 <= float(row[c]) <= 1 for c in bounded)
        assert math.isfinite(float(row['loss'])) and float(row['loss']) >= 0
        assert int(row['data_seed']) == 11 and int(row['n']) == 1024
        assert float(row['lr']) == .03 and int(row['steps']) == 500
    for k in expected:
        assert {s for *run, s in points if tuple(run) == k} == set(range(0, 501, 25))
    for row in final:
        assert all(row[c] == points[key(row) + (500,)][c] for c in history[0])
        assert float(row['accuracy']) == 1
        if row['experiment'] != 'ordinary':
            for c in ['initial_preactivation_max_error', 'initial_logit_max_error',
                      'initial_gradient_scaling_max_error']:
                assert float(row[c]) < 1e-5
        if row['experiment'] == 'compensated_sgd':
            assert float(row['final_logit_max_error_vs_alpha1_reference']) < 1e-4
    for row in history:
        if row['experiment'] == 'compensated_sgd':
            reference = points[('ordinary', int(row['seed']), 1., int(row['step']))]
            for c in bounded + ['loss']:
                assert np.isclose(float(row[c]), float(reference[c]), atol=2e-6, rtol=1e-5)

    baseline, baseline_history = read('02_feature_only_summary.csv'), read('02_feature_only_history.csv')
    assert len(baseline) == 10 and len(baseline_history) == 210
    bkey = lambda r: (r['experiment'], int(r['seed']), int(r['step']))
    bpoints = {bkey(r): r for r in baseline_history}
    assert len(bpoints) == len(baseline_history)
    assert set(bpoints) == set(itertools.product(['simple_only', 'complex_only'], SEEDS, range(0, 501, 25)))
    for row in baseline:
        assert float(row['accuracy']) == 1
        assert all(row[c] == bpoints[bkey(row)][c] for c in row)
        assert int(row['steps']) == 500 and float(row['lr']) == .03
        assert int(row['data_seed']) == 11 and int(row['n']) == 1024 and int(row['width']) == 32
        assert float(row['alpha']) == 1
    for row in read('aggregated_research_summary.csv'):
        group = [r for r in final if r['experiment'] == row['experiment']
                 and float(r['alpha']) == float(row['alpha'])]
        for dest, src in [('reliance_mean', 'complex_reliance'), ('loss_mean', 'loss'),
                          ('accuracy_mean', 'accuracy')]:
            assert np.isclose(float(row[dest]), np.mean([float(r[src]) for r in group]))
        for dest, src in [('reliance_sd', 'complex_reliance'), ('loss_sd', 'loss')]:
            assert np.isclose(float(row[dest]), np.std([float(r[src]) for r in group], ddof=1))
    print('PASS saved data: complete grids, checkpoint/final equality, baseline budgets, '
          'metric ranges, aggregates, and compensated reference histories')


def notebook_syntax_checks():
    for path in sorted(ROOT.glob('*.ipynb')):
        nb = json.loads(path.read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells'], 1):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'{path.name}:cell{i}')
                assert not any(o.get('output_type') == 'error' for o in cell.get('outputs', [])), path
    print('PASS notebook syntax and saved error checks')


if __name__ == '__main__':
    torch.set_num_threads(1)
    mathematical_checks()
    saved_result_checks()
    notebook_syntax_checks()
