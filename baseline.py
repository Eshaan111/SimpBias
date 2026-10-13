"""Pre-training baseline and mathematical controls. No model training is performed."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SCALES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
SEEDS = [0, 1, 2, 3, 4]


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = 2 * y - 1
    u = rng.choice([-1, 1], n)
    xs = t.astype(np.float32)[:, None]
    xc = np.stack([u, u * t], axis=1).astype(np.float32)
    return torch.from_numpy(xs), torch.from_numpy(xc), torch.from_numpy(y.astype(np.float32))[:, None]


class MLP(nn.Module):
    def __init__(self, d, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x)


def inputs(xs, xc, alpha, feature):
    if feature == 'simple':
        return xs
    if feature == 'complex':
        return alpha * xc
    return torch.cat([xs, alpha * xc], dim=1)


def metrics(model, x, y, xs, xc, alpha, feature, seed):
    with torch.no_grad():
        z = model(x)
        p = torch.sigmoid(z)
        acc = ((p >= 0.5).float() == y).float().mean().item()
        out = {'accuracy': acc, 'bce': F.binary_cross_entropy_with_logits(z, y).item(),
               'mean_abs_logit': z.abs().mean().item()}
        if feature != 'joint':
            return out
        # Fixed seeded permutations, complex coordinates kept together.
        gen = torch.Generator().manual_seed(seed + 2026)
        idx_s = torch.randperm(len(y), generator=gen)
        idx_c = torch.randperm(len(y), generator=gen)
        perm_s = torch.cat([xs[idx_s], alpha * xc], dim=1)
        perm_c = torch.cat([xs, alpha * xc[idx_c]], dim=1)
        def accuracy(inp):
            return ((torch.sigmoid(model(inp)) >= 0.5).float() == y).float().mean().item()
        out['perm_simple_drop'] = acc - accuracy(perm_s)
        out['perm_complex_drop'] = acc - accuracy(perm_c)
        # Paired flips: only one feature group changes, so conflict is intentional.
        flip_s = torch.cat([-xs, alpha * xc], dim=1)
        flip_c = torch.cat([xs, alpha * xc * torch.tensor([-1., 1.])], dim=1)
        original_pred = z >= 0
        out['simple_flip_rate'] = ((model(flip_s) >= 0) != original_pred).float().mean().item()
        out['complex_flip_rate'] = ((model(flip_c) >= 0) != original_pred).float().mean().item()
        out['simple_probability_change'] = (p - torch.sigmoid(model(flip_s))).abs().mean().item()
        out['complex_probability_change'] = (p - torch.sigmoid(model(flip_c))).abs().mean().item()
        # Conflict sets balanced in both directions; agreement with complex rule.
        pred_fs = model(flip_s) >= 0
        pred_fc = model(flip_c) >= 0
        complex_original_label = y >= 0.5
        complex_flipped_label = ~complex_original_label
        out['conflict_complex_preference'] = (0.5 * (
            (pred_fs == complex_original_label).float().mean() +
            (pred_fc == complex_flipped_label).float().mean())).item()
        return out


def gradient_metrics(model, x, y, alpha, feature):
    model.zero_grad(set_to_none=True)
    loss = F.binary_cross_entropy_with_logits(model(x), y)
    loss.backward()
    grad = model.net[0].weight.grad.detach()
    if feature == 'joint':
        return {'grad_simple_norm': grad[:, :1].norm().item(),
                'grad_complex_norm': grad[:, 1:].norm().item(),
                'effective_complex_step_norm_per_lr': (alpha * grad[:, 1:]).norm().item()}
    return {'grad_simple_norm': grad.norm().item() if feature == 'simple' else None,
            'grad_complex_norm': grad.norm().item() if feature == 'complex' else None,
            'effective_complex_step_norm_per_lr': (alpha * grad).norm().item() if feature == 'complex' else None}


def build_model(feature, seed, width):
    torch.manual_seed(seed)
    return MLP(1 if feature == 'simple' else 2 if feature == 'complex' else 3, width)


def inverse_initialize(model, alpha):
    with torch.no_grad():
        model.net[0].weight[:, 1:] /= alpha


def control_checks(xs, xc, y, alpha, seed, width, lr=0.01):
    x_ref = inputs(xs, xc, 1, 'joint')
    x_scaled = inputs(xs, xc, alpha, 'joint')
    a = build_model('joint', seed, width)
    b = build_model('joint', seed, width)
    inverse_initialize(b, alpha)
    with torch.no_grad():
        initial_logit_error = (a(x_ref) - b(x_scaled)).abs().max().item()
    a.zero_grad(set_to_none=True)
    b.zero_grad(set_to_none=True)
    F.binary_cross_entropy_with_logits(a(x_ref), y).backward()
    F.binary_cross_entropy_with_logits(b(x_scaled), y).backward()
    ga = a.net[0].weight.grad.detach().clone()
    gb = b.net[0].weight.grad.detach().clone()
    grad_scale_error = (gb[:, 1:] - alpha * ga[:, 1:]).abs().max().item()
    # Manual single SGD step: matched simple learning rate, complex lr / alpha^2.
    with torch.no_grad():
        for pa, pb in zip(a.parameters(), b.parameters()):
            if pa is a.net[0].weight:
                pa -= lr * pa.grad
                pb[:, :1] -= lr * pb.grad[:, :1]
                pb[:, 1:] -= (lr / alpha**2) * pb.grad[:, 1:]
            else:
                pa -= lr * pa.grad
                pb -= lr * pb.grad
        step_logit_error = (a(x_ref) - b(x_scaled)).abs().max().item()
    return {'initial_max_logit_difference': initial_logit_error,
            'complex_gradient_scaling_max_difference': grad_scale_error,
            'compensated_step_max_logit_difference': step_logit_error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results')
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for seed in SEEDS[:args.seeds]:
        xs, xc, y = make_data(args.samples, seed + 10000)
        # Verify dataset construction, including perfect individual signals.
        assert torch.equal((xs > 0).float(), y)
        assert torch.equal((xc[:, :1] * xc[:, 1:] > 0).float(), y)
        for feature in ['simple', 'complex', 'joint']:
            for alpha in SCALES:
                for init in (['standard', 'inverse'] if feature == 'joint' else ['standard']):
                    model = build_model(feature, seed, args.width)
                    if init == 'inverse':
                        inverse_initialize(model, alpha)
                    x = inputs(xs, xc, alpha, feature)
                    record = {'seed': seed, 'feature': feature, 'alpha': alpha, 'initialization': init}
                    record.update(metrics(model, x, y, xs, xc, alpha, feature, seed))
                    record.update(gradient_metrics(model, x, y, alpha, feature))
                    rows.append(record)
        for alpha in SCALES:
            checks.append({'seed': seed, 'alpha': alpha, **control_checks(xs, xc, y, alpha, seed, args.width)})
    with (outdir / 'pretrain_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row.keys())))
        writer.writeheader()
        writer.writerows(rows)
    with (outdir / 'control_checks.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)
    with (outdir / 'metadata.json').open('w') as f:
        json.dump({'stage': 'PRETRAIN_ONLY', 'training_performed': False, 'seeds': SEEDS[:args.seeds],
                   'scales': SCALES, 'samples_per_seed': args.samples, 'hidden_width': args.width,
                   'optimizer_for_mathematical_control': 'plain SGD, manual single step'}, f, indent=2)
    print('Wrote', outdir / 'pretrain_metrics.csv', 'and', outdir / 'control_checks.csv')
    print('Max compensated-step logit discrepancy:', max(x['compensated_step_max_logit_difference'] for x in checks))


if __name__ == '__main__':
    main()
