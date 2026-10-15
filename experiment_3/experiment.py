"""Shared, readable functions for the four Experiment 3 notebooks.

No imports from the earlier experiments. Inputs are never standardized.
"""
from pathlib import Path
import csv
import json
import math

import numpy as np
import torch
from torch import nn

DATASETS = ['C0_original', 'C1_zero', 'C2_noise']
CONDITIONS = ['ordinary', 'inverse_init', 'compensated_sgd']
ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS = [42, 43, 44, 45, 46]
INITIAL_TOLERANCE = 1e-6
TRAJECTORY_TOLERANCE = 1e-4


def make_base(n, seed, conflict=False):
    """Return [simple, noise, complex_1, complex_2] before scaling.

    The predictive coordinates exactly reuse the previous continuous sampler.
    Noise uses a separate generator, so adding it cannot change those samples.
    Conflict labels are complex-implied labels, not OOD ground truth.
    """
    rng = np.random.default_rng(seed)
    if conflict:
        if n <= 0 or n % 4:
            raise ValueError('Conflict count must be a positive multiple of four.')
        signs = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (n // 4, 1))
        rng.shuffle(signs)
        simple_sign, u = signs.T
        target_sign = -simple_sign
        y = (target_sign + 1) / 2
    else:
        y = rng.integers(0, 2, n)
        target_sign = simple_sign = 2 * y - 1
        u = rng.choice([-1, 1], n)
    magnitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    predictive = np.column_stack([simple_sign, u, u * target_sign]) * magnitudes
    noise = np.random.default_rng(seed + 100_000).normal(size=n)
    x = np.column_stack([predictive[:, 0], noise, predictive[:, 1:]])
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


def make_splits():
    return {name: make_base(n, seed, conflict=(name == 'conflicts'))
            for name, n, seed in [('train', 1024, 11), ('validation', 1024, 12),
                                  ('test', 2048, 13), ('conflicts', 2048, 14)]}


def complex_columns(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    return slice(1, 3) if dataset == 'C0_original' else slice(2, 4)


def input_data(base, dataset, alpha=1.0):
    """Select a view of the SAME base samples, then scale complex columns only."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive.')
    cols = complex_columns(dataset)
    x, y = base
    x = x[:, [0, 2, 3]].clone() if dataset == 'C0_original' else x.clone()
    if dataset == 'C1_zero':
        x[:, 1] = 0
    x[:, cols] *= alpha
    return x, y


class MLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_model(seed, dataset='C0_original', alpha=1.0, condition='ordinary'):
    """Copy shared parameters rather than relying on equal random seeds.

    The nuisance column has its own seed and the same Uniform bound as the
    three-input reference columns. We intentionally do not redraw shared
    weights with the default four-input initialization bound.
    """
    cols = complex_columns(dataset)
    if condition not in CONDITIONS or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Invalid condition or scale.')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = MLP(3)
        if dataset == 'C0_original':
            model = reference
        else:
            model = MLP(4)
            generator = torch.Generator().manual_seed(seed + 200_000)
            with torch.no_grad():
                model.fc1.weight[:, 0] = reference.fc1.weight[:, 0]
                model.fc1.weight[:, 2:4] = reference.fc1.weight[:, 1:3]
                model.fc1.weight[:, 1].uniform_(-1 / math.sqrt(3), 1 / math.sqrt(3), generator=generator)
                model.fc1.bias.copy_(reference.fc1.bias)
                model.fc2.load_state_dict(reference.fc2.state_dict())
    if condition != 'ordinary':
        with torch.no_grad():
            model.fc1.weight[:, cols] /= alpha
    return model


@torch.no_grad()
def classification_metrics(model, data):
    x, y = data
    z = model(x)
    return {'loss': nn.functional.binary_cross_entropy_with_logits(z, y).item(),
            'accuracy': ((z >= 0).float() == y).float().mean().item()}


@torch.no_grad()
def conflict_metrics(model, data):
    x, complex_y = data
    complex_agreement = ((model(x) >= 0).float() == complex_y).float().mean().item()
    # Binary contradictory rules: these two agreements sum to one.
    return {'complex_conflict_agreement': complex_agreement,
            'simple_conflict_agreement': 1 - complex_agreement}


@torch.no_grad()
def counterfactual_metrics(model, x):
    simple, complex_ = x.clone(), x.clone()
    simple[:, 0] *= -1
    complex_[:, -1] *= -1  # One complex sign reverses XNOR; noise is unchanged.
    z, zs, zc = model(x), model(simple), model(complex_)
    return {'simple_flip_rate': ((z >= 0) != (zs >= 0)).float().mean().item(),
            'complex_flip_rate': ((z >= 0) != (zc >= 0)).float().mean().item(),
            'simple_probability_change': (z.sigmoid() - zs.sigmoid()).abs().mean().item(),
            'complex_probability_change': (z.sigmoid() - zc.sigmoid()).abs().mean().item()}


def initial_checks(reference, model, original, scaled, dataset, alpha):
    """Check matching preactivations/logits and the chain-rule gradient factor."""
    cols = complex_columns(dataset)
    x, y = original
    xa, _ = scaled
    with torch.no_grad():
        pre = (reference.fc1(x) - model.fc1(xa)).abs().max().item()
        logits = (reference(x) - model(xa)).abs().max().item()
    reference.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    nn.functional.binary_cross_entropy_with_logits(reference(x), y).backward()
    nn.functional.binary_cross_entropy_with_logits(model(xa), y).backward()
    grad = (model.fc1.weight.grad[:, cols] - alpha * reference.fc1.weight.grad[:, cols]).abs().max().item()
    reference.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    return {'initial_preactivation_error': pre, 'initial_logit_error': logits,
            'initial_gradient_scaling_error': grad}


@torch.no_grad()
def snapshot(model, check_x, dataset, alpha):
    """Effective parameters include the nuisance column when it exists."""
    w = model.fc1.weight.detach().clone()
    w[:, complex_columns(dataset)] *= alpha
    rest = [model.fc1.bias, model.fc2.weight, model.fc2.bias]
    shared_w = w if dataset == 'C0_original' else w[:, [0, 2, 3]]
    return {'logits': model(check_x).detach().clone(),
            'parameters': torch.cat([w.flatten()] + [v.detach().flatten() for v in rest]),
            'shared_parameters': torch.cat([shared_w.flatten()] + [v.detach().flatten() for v in rest])}


def train_model(model, train, validation, steps=500, lr=0.03, dataset=None,
                alpha=1.0, condition='ordinary', conflicts=None, on_log=None):
    """Only training labels enter backward(); validation never selects a model.

    Snapshots compare validation AND conflict logits at every saved checkpoint.
    No test examples are used to select or monitor the optimization trajectory.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    history, snapshots = [], {}
    x, y = train
    for step in range(steps + 1):
        if step % 25 == 0 or step == steps:
            row = {'step': step}
            for name, data in [('train', train), ('validation', validation)]:
                row.update({f'{name}_{k}': v for k, v in classification_metrics(model, data).items()})
            if conflicts is not None:
                row.update(conflict_metrics(model, conflicts))
                row.update({f'validation_{k}': v for k, v in counterfactual_metrics(model, validation[0]).items()})
                check_x = torch.cat([validation[0], conflicts[0]])
                snapshots[step] = snapshot(model, check_x, dataset, alpha)
            history.append(row)
            if on_log is not None:
                on_log(history)
        if step == steps:
            break
        opt.zero_grad()
        criterion(model(x), y).backward()
        if condition == 'compensated_sgd':
            # Backpropagation contributes alpha; effective weights contribute
            # another alpha. Cancel both after backward and before opt.step.
            model.fc1.weight.grad[:, complex_columns(dataset)] /= alpha**2
        opt.step()
        if not all(torch.isfinite(v).all() for v in model.parameters()):
            raise FloatingPointError(f'Nonfinite parameter after update {step + 1}')
    return history, snapshots


def compare_snapshots(current, reference, parameter_key='parameters'):
    return [{'step': step,
             'logit_error': (snap['logits'] - reference[step]['logits']).abs().max().item(),
             'parameter_error': (snap[parameter_key] - reference[step][parameter_key]).abs().max().item()}
            for step, snap in current.items()]


def save_rows(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        return
    # Replace a complete CSV snapshot; never append rows from a previous run.
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def save_splits(folder, splits):
    for name, (x, y) in splits.items():
        rows = []
        for coordinates, label in zip(x.tolist(), y[:, 0].tolist()):
            row = dict(zip(['simple', 'noise', 'complex_1', 'complex_2'], coordinates))
            if name == 'conflicts':
                row.update(complex_implied_label=int(label), simple_implied_label=1 - int(label))
            else:
                row['label'] = int(label)
            rows.append(row)
        save_rows(Path(folder) / f'{name}_data.csv', rows)


def settings(quick=False):
    return {'seeds': [42] if quick else SEEDS, 'alphas': [1.0, 5.0] if quick else ALPHAS,
            'steps': 25 if quick else 500, 'lr': 0.03, 'quick': quick}



def baseline_data(base, group):
    x, y = base
    if group == 'C0_simple': return x[:, :1], y
    if group == 'C1_simple':
        x = x[:, :2].clone()
        x[:, 1] = 0
        return x, y
    if group == 'C2_simple': return x[:, :2], y
    if group == 'shared_complex': return x[:, 2:4], y
    raise ValueError(group)


def make_baseline(seed, group):
    # Match C0/C1/C2 simple baselines to the SAME one-input starting weights.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = MLP(2 if group == 'shared_complex' else 1)
        if group in ['C0_simple', 'shared_complex']:
            return reference
        model = MLP(2)
        with torch.no_grad():
            model.fc1.weight[:, 0] = reference.fc1.weight[:, 0]
            model.fc1.weight[:, 1].uniform_(-1, 1, generator=torch.Generator().manual_seed(seed + 200_000))
            model.fc1.bias.copy_(reference.fc1.bias)
            model.fc2.load_state_dict(reference.fc2.state_dict())
    return model
