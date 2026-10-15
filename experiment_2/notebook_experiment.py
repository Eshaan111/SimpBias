"""Small shared functions for the continuous-XNOR notebooks only."""
import csv
import json

import numpy as np
import torch
from torch import nn


def make_data(n, seed):
    """Generate unscaled continuous inputs whose two rules agree."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = 2 * y - 1
    u = rng.choice([-1, 1], n)
    # Independent positive magnitudes preserve the sign-based label rules.
    magnitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    x = np.column_stack([t, u, u * t]) * magnitudes
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


def make_conflicts(n, seed=14):
    """Balance all four sign configurations, but make the two rules disagree."""
    if n <= 0 or n % 4:
        raise ValueError('Use a positive multiple of 4 conflict probes.')
    rng = np.random.default_rng(seed)
    combinations = np.tile([[-1, -1], [-1, 1], [1, -1], [1, 1]], (n // 4, 1))
    rng.shuffle(combinations)
    simple_sign, u = combinations.T
    complex_sign = -simple_sign
    magnitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    x = np.column_stack([simple_sign, u, u * complex_sign]) * magnitudes
    # This is the complex rule's implied label, not an OOD ground-truth label.
    return torch.tensor(x, dtype=torch.float32), torch.tensor((complex_sign + 1) / 2, dtype=torch.float32).reshape(-1, 1)


def make_splits():
    # Distinct seeds supply independent coordinates. Every notebook regenerates
    # these same splits, so comparisons use identical underlying samples.
    return {
        'train': make_data(1024, 11),
        'validation': make_data(1024, 12),
        'test': make_data(2048, 13),
        'conflicts': make_conflicts(2048, 14),
    }


def scale_data(data, alpha):
    """Copy a split and scale only its complex columns; never resample."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive.')
    x, y = data
    scaled = x.clone()
    scaled[:, 1:3] *= alpha
    return scaled, y


class MLP(nn.Module):
    def __init__(self, inputs=3):
        super().__init__()
        self.fc1 = nn.Linear(inputs, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def make_model(seed, inputs=3):
    torch.manual_seed(seed)
    return MLP(inputs)


def inverse_initialize(model, reference, alpha):
    model.load_state_dict(reference.state_dict())
    with torch.no_grad():
        # (alpha*x) * (weight/alpha) preserves the initial weighted sum.
        model.fc1.weight[:, 1:3] /= alpha


@torch.no_grad()
def classification_metrics(model, data):
    x, y = data
    logits = model(x)
    return {
        'loss': nn.functional.binary_cross_entropy_with_logits(logits, y).item(),
        'accuracy': ((logits >= 0).float() == y).float().mean().item(),
    }


@torch.no_grad()
def conflict_agreement(model, data):
    x, complex_y = data
    return ((model(x) >= 0).float() == complex_y).float().mean().item()


@torch.no_grad()
def counterfactual_metrics(model, x):
    simple_flipped = x.clone()
    simple_flipped[:, 0] *= -1
    complex_flipped = x.clone()
    complex_flipped[:, 2] *= -1  # One sign reverses XNOR; two would not.
    z = model(x)
    zs = model(simple_flipped)
    zc = model(complex_flipped)
    return {
        'simple_flip_rate': ((z >= 0) != (zs >= 0)).float().mean().item(),
        'complex_flip_rate': ((z >= 0) != (zc >= 0)).float().mean().item(),
        'simple_probability_change': (z.sigmoid() - zs.sigmoid()).abs().mean().item(),
        'complex_probability_change': (z.sigmoid() - zc.sigmoid()).abs().mean().item(),
    }


def initial_checks(reference, scaled, original_data, scaled_data, alpha):
    x, y = original_data
    xa, _ = scaled_data
    with torch.no_grad():
        z_error = (reference.fc1(x) - scaled.fc1(xa)).abs().max().item()
        logit_error = (reference(x) - scaled(xa)).abs().max().item()
    reference.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    criterion = nn.BCEWithLogitsLoss()
    criterion(reference(x), y).backward()
    criterion(scaled(xa), y).backward()
    grad_error = (scaled.fc1.weight.grad[:, 1:3]
                  - alpha * reference.fc1.weight.grad[:, 1:3]).abs().max().item()
    reference.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    return {'initial_preactivation_max_error': z_error,
            'initial_logit_max_error': logit_error,
            'initial_gradient_scaling_max_error': grad_error}


def train_model(model, train_data, validation_data, steps=500, lr=0.03,
                alpha=1.0, compensate=False, conflicts=None, on_log=None):
    """Full-batch SGD. on_log optionally updates a notebook's live display."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    x_train, y_train = train_data
    history = []
    for step in range(steps + 1):
        if step % 25 == 0 or step == steps:
            row = {'step': step}
            for name, data in [('train', train_data), ('validation', validation_data)]:
                row.update({f'{name}_{k}': v for k, v in classification_metrics(model, data).items()})
            if conflicts is not None:
                row['complex_conflict_agreement'] = conflict_agreement(model, conflicts)
                row.update({f'validation_{k}': v for k, v in counterfactual_metrics(model, validation_data[0]).items()})
            history.append(row)
            if on_log is not None:
                on_log(history)
        if step == steps:
            break
        optimizer.zero_grad()
        criterion(model(x_train), y_train).backward()
        if compensate:
            # One alpha comes from the gradient; another converts a stored
            # weight update into an effective update. Cancel both before SGD.
            model.fc1.weight.grad[:, 1:3] /= alpha**2
        optimizer.step()
    return history


def save_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2), encoding='utf-8')


def save_splits(folder, splits):
    for name, (x, y) in splits.items():
        rows = []
        for coordinates, label in zip(x.tolist(), y[:, 0].tolist()):
            row = dict(zip(['simple', 'complex_1', 'complex_2'], coordinates))
            if name == 'conflicts':
                row.update(complex_implied_label=int(label), simple_implied_label=1 - int(label))
            else:
                row['label'] = int(label)
            rows.append(row)
        save_rows(folder / f'{name}_data.csv', rows)
