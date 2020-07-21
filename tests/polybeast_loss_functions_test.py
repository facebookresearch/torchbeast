# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for polybeast loss functions implementation."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F
from torchbeast import polybeast_learner as polybeast


def _softmax(logits):
    """Applies softmax non-linearity on inputs."""
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _softmax_grad(logits):
    """Compute the gradient of softmax function."""
    s = np.expand_dims(_softmax(logits), 0)
    return s.T * (np.eye(s.size) - s)


def assert_allclose(actual, desired):
    return np.testing.assert_allclose(actual, desired, rtol=1e-06, atol=1e-05)


class ComputeBaselineLossTest(unittest.TestCase):
    def setUp(self):
        # Floating point constants are randomly generated.
        self.advantages = np.array([1.4, 3.43, 5.2, 0.33])

    def test_compute_baseline_loss(self):
        ground_truth_value = 0.5 * np.sum(self.advantages ** 2)
        assert_allclose(
            ground_truth_value,
            polybeast.compute_baseline_loss(torch.from_numpy(self.advantages)),
        )

    def test_compute_baseline_loss_grad(self):
        advantages_tensor = torch.from_numpy(self.advantages)
        advantages_tensor.requires_grad_()
        calculated_value = polybeast.compute_baseline_loss(advantages_tensor)
        calculated_value.backward()

        # Manually computed gradients:
        # 0.5 * d(xˆ2)/dx == x
        # hence the expected gradient is the same as self.advantages.
        assert_allclose(advantages_tensor.grad, self.advantages)


class ComputeEntropyLossTest(unittest.TestCase):
    def setUp(self):
        # Floating point constants are randomly generated.
        self.logits = np.array([0.0012, 0.321, 0.523, 0.109, 0.416])

    def test_compute_entropy_loss(self):
        # Calculate entropy with:
        # H(s) = - sum(prob(x) * ln(prob(x)) for each x in s)
        softmax_logits = _softmax(self.logits)
        ground_truth_value = np.sum(softmax_logits * np.log(softmax_logits))
        calculated_value = polybeast.compute_entropy_loss(torch.from_numpy(self.logits))

        assert_allclose(ground_truth_value, calculated_value)

    def test_compute_entropy_loss_grad(self):
        logits_tensor = torch.from_numpy(self.logits)
        logits_tensor.requires_grad_()
        calculated_value = polybeast.compute_entropy_loss(logits_tensor)
        calculated_value.backward()

        expected_grad = np.matmul(
            np.ones_like(self.logits),
            np.matmul(
                np.diag(1 + np.log(_softmax(self.logits))), _softmax_grad(self.logits)
            ),
        )

        assert_allclose(logits_tensor.grad, expected_grad)


class ComputePolicyGradientLossTest(unittest.TestCase):
    def setUp(self):
        # Floating point constants are randomly generated.
        self.logits = np.array(
            [
                [
                    [0.206, 0.738, 0.125, 0.484, 0.332],
                    [0.168, 0.504, 0.523, 0.496, 0.626],
                    [0.236, 0.186, 0.627, 0.441, 0.533],
                ],
                [
                    [0.015, 0.904, 0.583, 0.651, 0.855],
                    [0.811, 0.292, 0.061, 0.597, 0.590],
                    [0.999, 0.504, 0.464, 0.077, 0.143],
                ],
            ]
        )
        self.actions = np.array([[3, 0, 1], [4, 2, 2]])
        self.advantages = np.array([[1.4, 0.31, 0.75], [2.1, 1.5, 0.03]])

    def test_compute_policy_gradient_loss(self):
        T, B, N = self.logits.shape

        # Calculate the the cross entropy loss, with the formula:
        # loss = -sum_over_j(y_j * log(p_j))
        # Where:
        # - `y_j` is whether the action corrisponding to index j has been taken or not,
        #   (hence y is a one-hot-array of size == number of actions).
        # - `p_j` is the value of the sofmax logit corresponding to the jth action.
        # In our implementation, we also multiply for the advantages.
        labels = F.one_hot(torch.from_numpy(self.actions), num_classes=N).numpy()
        cross_entropy_loss = -labels * np.log(_softmax(self.logits))
        ground_truth_value = np.sum(
            cross_entropy_loss * self.advantages.reshape(T, B, 1)
        )

        calculated_value = polybeast.compute_policy_gradient_loss(
            torch.from_numpy(self.logits),
            torch.from_numpy(self.actions),
            torch.from_numpy(self.advantages),
        )
        assert_allclose(ground_truth_value, calculated_value.item())

    def test_compute_policy_gradient_loss_grad(self):
        T, B, N = self.logits.shape

        logits_tensor = torch.from_numpy(self.logits)
        logits_tensor.requires_grad_()

        calculated_value = polybeast.compute_policy_gradient_loss(
            logits_tensor,
            torch.from_numpy(self.actions),
            torch.from_numpy(self.advantages),
        )

        self.assertSequenceEqual(calculated_value.shape, [])
        calculated_value.backward()

        # The gradient of the cross entropy loss function for the jth logit
        # can be expressed as:
        # p_j - y_j
        # where:
        # - `p_j` is the value of the softmax logit corresponding to the jth action.
        # - `y_j` is whether the action corrisponding to index j has been taken,
        #   (hence y is a one-hot-array of size == number of actions).
        # In our implementation, we also multiply for the advantages.
        softmax = _softmax(self.logits)
        labels = F.one_hot(torch.from_numpy(self.actions), num_classes=N).numpy()
        expected_grad = (softmax - labels) * self.advantages.reshape(T, B, 1)

        assert_allclose(logits_tensor.grad, expected_grad)

    def test_compute_policy_gradient_loss_grad_flow(self):
        logits_tensor = torch.from_numpy(self.logits)
        logits_tensor.requires_grad_()
        advantages_tensor = torch.from_numpy(self.advantages)
        advantages_tensor.requires_grad_()

        loss = polybeast.compute_policy_gradient_loss(
            logits_tensor, torch.from_numpy(self.actions), advantages_tensor
        )
        loss.backward()

        self.assertIsNotNone(logits_tensor.grad)
        self.assertIsNone(advantages_tensor.grad)


if __name__ == "__main__":
    unittest.main()
