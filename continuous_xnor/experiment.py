"""Data, model, and measurements for the independent continuous-XNOR test."""
import numpy as np
import torch
from torch import nn


def make_data(n, seed):
    """Return UNSCALED samples. Both feature groups reveal the same label."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = 2 * y - 1
    u = rng.choice([-1, 1], n)

    # Positive magnitudes vary independently, so signs still define the rules.
    # Keep them away from zero to leave a margin around the decision axes.
    magnitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    signs = np.column_stack([t, u, u * t])
    x = torch.tensor(signs * magnitudes, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return x, y


def make_conflicts(n, seed):
    """Return contradictory inputs and their COMPLEX-implied classes.

    These classes describe the diagnostic rule, not a ground-truth OOD label.
    Each of the four (simple sign, nuisance sign) combinations occurs equally.
    """
    if n <= 0 or n % 4 != 0:
        raise ValueError('The number of conflict probes must be a positive multiple of 4.')
    rng = np.random.default_rng(seed)
    combinations = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    combinations = np.tile(combinations, (n // 4, 1))
    rng.shuffle(combinations)
    simple_sign, u = combinations.T
    complex_sign = -simple_sign  # Make the two rules disagree deliberately.
    signs = np.column_stack([simple_sign, u, u * complex_sign])
    magnitudes = rng.uniform(0.5, 1.5, size=(n, 3))
    x = torch.tensor(signs * magnitudes, dtype=torch.float32)
    complex_y = torch.tensor((complex_sign + 1) / 2, dtype=torch.float32).reshape(-1, 1)
    return x, complex_y


def scale_complex(x, alpha):
    """Scale only the two complex coordinates; never change the base samples."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('alpha must be finite and positive.')
    scaled = x.clone()
    scaled[:, 1:3] *= alpha
    return scaled


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
    # load_state_dict copies every parameter. Only the complex columns change.
    model.load_state_dict(reference.state_dict())
    with torch.no_grad():
        model.fc1.weight[:, 1:3] /= alpha


@torch.no_grad()
def classification_metrics(model, x, y):
    logits = model(x)
    return {
        'loss': nn.functional.binary_cross_entropy_with_logits(logits, y).item(),
        'accuracy': ((logits >= 0).float() == y).float().mean().item(),
    }


@torch.no_grad()
def conflict_agreement(model, conflict_data):
    x, complex_y = conflict_data
    prediction = (model(x) >= 0).float()
    return (prediction == complex_y).float().mean().item()


@torch.no_grad()
def counterfactual_metrics(model, x):
    # Positive magnitudes do not alter the intervention: one sign still flips.
    simple_flipped = x.clone()
    simple_flipped[:, 0] *= -1
    complex_flipped = x.clone()
    complex_flipped[:, 2] *= -1
    original = model(x)
    simple = model(simple_flipped)
    complex_ = model(complex_flipped)
    return {
        'simple_flip_rate': ((original >= 0) != (simple >= 0)).float().mean().item(),
        'complex_flip_rate': ((original >= 0) != (complex_ >= 0)).float().mean().item(),
        'simple_probability_change': (original.sigmoid() - simple.sigmoid()).abs().mean().item(),
        'complex_probability_change': (original.sigmoid() - complex_.sigmoid()).abs().mean().item(),
    }


def initial_checks(reference, scaled, x, scaled_x, y, alpha):
    """Check the matched forward function and the alpha-scaled gradient."""
    with torch.no_grad():
        z_error = (reference.fc1(x) - scaled.fc1(scaled_x)).abs().max().item()
        logit_error = (reference(x) - scaled(scaled_x)).abs().max().item()
    reference.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    criterion = nn.BCEWithLogitsLoss()
    criterion(reference(x), y).backward()
    criterion(scaled(scaled_x), y).backward()
    grad_error = (scaled.fc1.weight.grad[:, 1:3]
                  - alpha * reference.fc1.weight.grad[:, 1:3]).abs().max().item()
    reference.zero_grad(set_to_none=True)
    scaled.zero_grad(set_to_none=True)
    return {
        'initial_preactivation_max_error': z_error,
        'initial_logit_max_error': logit_error,
        'initial_gradient_scaling_max_error': grad_error,
    }


def train_model(model, train_data, validation_data, steps=500, lr=0.03,
                alpha=1.0, compensate=False, conflict_data=None, record_every=25):
    """Full-batch plain SGD. Test samples are never used in this function."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    x_train, y_train = train_data
    x_validation, _ = validation_data
    history = []

    for step in range(steps + 1):
        if step % record_every == 0 or step == steps:
            row = {'step': step}
            for name, data in [('train', train_data), ('validation', validation_data)]:
                metrics = classification_metrics(model, *data)
                row.update({f'{name}_{key}': value for key, value in metrics.items()})
            if conflict_data is not None:
                row['complex_conflict_agreement'] = conflict_agreement(model, conflict_data)
                metrics = counterfactual_metrics(model, x_validation)
                row.update({f'validation_{key}': value for key, value in metrics.items()})
            history.append(row)
        if step == steps:
            break

        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        if compensate:
            # Gradient gains alpha; converting to effective weights gains alpha
            # again. Dividing by alpha^2 cancels BOTH factors before the step.
            model.fc1.weight.grad[:, 1:3] /= alpha**2
        optimizer.step()
    return history
