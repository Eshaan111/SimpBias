"""Small mathematical checks, not the full research sweep: python test_experiment.py."""
import unittest

import torch

from experiment import (
    make_data, make_conflicts, scale_complex, make_model, inverse_initialize,
    conflict_agreement, counterfactual_metrics, initial_checks, train_model,
)


class ContinuousXNORTests(unittest.TestCase):
    def test_continuous_data_and_labels(self):
        x, y = make_data(256, seed=11)
        self.assertEqual(len(torch.unique(x, dim=0)), 256)
        self.assertTrue(torch.equal((x[:, :1] > 0).float(), y))
        self.assertTrue(torch.equal((x[:, 1:2] * x[:, 2:3] > 0).float(), y))
        self.assertTrue(bool(((x.abs() >= 0.5) & (x.abs() <= 1.5)).all()))
        for alpha in [0.1, 0.5, 1, 2, 5, 10]:
            scaled = scale_complex(x, alpha)
            self.assertTrue(torch.equal((scaled[:, 1:2] * scaled[:, 2:3] > 0).float(), y))
            torch.testing.assert_close(scaled[:, 0], x[:, 0])
        torch.testing.assert_close(x, make_data(256, seed=11)[0])

    def test_splits_have_distinct_samples(self):
        train = make_data(128, seed=11)[0]
        test = make_data(128, seed=13)[0]
        self.assertEqual(len(torch.unique(torch.cat([train, test]), dim=0)), 256)

    def test_conflicts_are_balanced_and_contradictory(self):
        x, complex_y = make_conflicts(128, seed=14)
        self.assertTrue(torch.equal((x[:, 1:2] * x[:, 2:3] > 0).float(), complex_y))
        self.assertTrue(torch.equal((x[:, :1] > 0).float(), 1 - complex_y))
        for simple_sign in [-1, 1]:
            for nuisance_sign in [-1, 1]:
                count = ((x[:, 0].sign() == simple_sign) & (x[:, 1].sign() == nuisance_sign)).sum()
                self.assertEqual(int(count), 32)
        torch.testing.assert_close(x, make_conflicts(128, seed=14)[0])

    def test_interventions_against_ideal_rules(self):
        x, _ = make_data(128, seed=13)
        conflicts = make_conflicts(128, seed=14)
        simple_rule = lambda inputs: inputs[:, :1]
        complex_rule = lambda inputs: inputs[:, 1:2] * inputs[:, 2:3]
        self.assertEqual(conflict_agreement(simple_rule, conflicts), 0)
        self.assertEqual(conflict_agreement(complex_rule, conflicts), 1)
        for rule, simple_rate, complex_rate in [(simple_rule, 1, 0), (complex_rule, 0, 1)]:
            metrics = counterfactual_metrics(rule, x)
            self.assertEqual(metrics['simple_flip_rate'], simple_rate)
            self.assertEqual(metrics['complex_flip_rate'], complex_rate)
            expected = (rule(x).sigmoid() - rule(x * torch.tensor([-1., 1., 1.])).sigmoid()).abs().mean()
            self.assertAlmostEqual(metrics['simple_probability_change'], expected.item())

    def test_inverse_initialization_and_gradient_scaling(self):
        x, y = make_data(128, seed=11)
        for alpha in [0.1, 0.5, 1, 2, 5, 10]:
            reference, model = make_model(42), make_model(42)
            inverse_initialize(model, reference, alpha)
            checks = initial_checks(reference, model, x, scale_complex(x, alpha), y, alpha)
            self.assertTrue(all(error < 1e-5 for error in checks.values()), checks)
            torch.testing.assert_close(model.fc1.weight[:, 0], reference.fc1.weight[:, 0])
            torch.testing.assert_close(model.fc1.bias, reference.fc1.bias)
            torch.testing.assert_close(model.fc2.weight, reference.fc2.weight)

    def test_compensated_training_matches_reference(self):
        train_data, validation_data = make_data(128, 11), make_data(128, 12)
        test_x, _ = make_data(128, 13)
        for alpha in [0.1, 0.5, 1, 2, 5, 10]:
            reference, model = make_model(42), make_model(42)
            inverse_initialize(model, reference, alpha)
            scaled_train = (scale_complex(train_data[0], alpha), train_data[1])
            scaled_validation = (scale_complex(validation_data[0], alpha), validation_data[1])
            train_model(reference, train_data, validation_data, steps=8)
            train_model(model, scaled_train, scaled_validation, steps=8, alpha=alpha, compensate=True)
            torch.testing.assert_close(reference(test_x), model(scale_complex(test_x, alpha)),
                                       atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
