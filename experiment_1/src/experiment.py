"""Small shared utilities for the simplicity-bias notebooks."""
import copy
import random
import numpy as np
import torch
from torch import nn


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_data(n=1024, seed=42, alpha=1.0):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = 2 * y - 1
    u = rng.choice([-1, 1], n)
    x = np.column_stack([t, alpha*u, alpha*u*t]).astype(np.float32)
    return torch.tensor(x), torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class SmallMLP(nn.Module):
    def __init__(self, in_features=3, width=32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(width, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def make_model(seed=42, in_features=3, width=32):
    seed_everything(seed)
    return SmallMLP(in_features, width)


def inverse_initialize(model, reference, alpha):
    model.load_state_dict(copy.deepcopy(reference.state_dict()))
    with torch.no_grad():
        model.fc1.weight[:, 1:3] /= alpha
    return model


def accuracy(logits, y):
    return ((logits >= 0).float() == y).float().mean().item()


def conflict_reliance(model, alpha=1.0):
    # Contradictory simple and complex labels: out-of-distribution diagnostic.
    xs = []
    complex_labels = []
    for ts in (-1, 1):
        for u in (-1, 1):
            tc = -ts
            xs.append([ts, alpha*u, alpha*u*tc])
            complex_labels.append((tc+1)//2)
    x = torch.tensor(xs, dtype=torch.float32)
    y_complex = torch.tensor(complex_labels, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        return accuracy(model(x), y_complex)



def counterfactual_sensitivity(model, x):
    """Evaluate paired feature flips on normal examples (OOD interventions).

    Returns hard prediction-flip rates and mean absolute probability changes.
    Complex flip negates ONE complex coordinate, reversing its implied label.
    """
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError('Expected joint inputs with shape [N, 3].')
    x_simple = x.clone()
    x_simple[:, 0] *= -1
    x_complex = x.clone()
    x_complex[:, 2] *= -1
    with torch.no_grad():
        z = model(x)
        z_s = model(x_simple)
        z_c = model(x_complex)
        p = torch.sigmoid(z)
        return {
            'simple_flip_rate': ((z >= 0) != (z_s >= 0)).float().mean().item(),
            'complex_flip_rate': ((z >= 0) != (z_c >= 0)).float().mean().item(),
            'simple_probability_change': (p - torch.sigmoid(z_s)).abs().mean().item(),
            'complex_probability_change': (p - torch.sigmoid(z_c)).abs().mean().item(),
        }


def train(model, x, y, steps=500, lr=0.03, alpha=1.0, compensate=False, record_every=25):
    """Full-batch plain SGD. No momentum, weight decay, or clipping.

    With compensate=True, rescale ONLY the two complex columns' gradients.
    This is equivalent to a per-column learning rate lr / alpha**2.
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for step in range(steps + 1):
        if step % record_every == 0 or step == steps:
            with torch.no_grad():
                logits = model(x)
                history.append(dict(step=step, loss=loss_fn(logits, y).item(),
                                    accuracy=accuracy(logits, y),
                                    complex_reliance=conflict_reliance(model, alpha),
                                    **counterfactual_sensitivity(model, x)))
        if step == steps:
            break
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        if compensate:
            with torch.no_grad():
                model.fc1.weight.grad[:, 1:3] /= alpha**2
        opt.step()
    return history


def plot_histories(histories, key='complex_reliance'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4))
    for label, history in histories.items():
        plt.plot([r['step'] for r in history], [r[key] for r in history], label=label)
    plt.xlabel('SGD step')
    plt.ylabel(key.replace('_', ' '))
    plt.legend()
    plt.grid(alpha=0.2)
    plt.show()


def save_rows(rows, path):
    import csv
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
