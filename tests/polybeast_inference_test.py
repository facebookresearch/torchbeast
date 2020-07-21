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
"""Tests for polybeast inference implementation."""

import unittest
import warnings
from unittest import mock

import torch
from torchbeast import polybeast_learner as polybeast


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.unroll_length = 1  # Inference called for every step.
        self.batch_size = 4  # Arbitrary.
        self.frame_dimension = 84  # Has to match what expected by the model.
        self.num_actions = 6  # Specific to each environment.
        self.num_channels = 4  # Has to match with the first conv layer of the net.
        self.core_output_size = 256  # Has to match what expected by the model.
        self.num_lstm_layers = 1  # As in the model.

        self.frame = torch.ones(
            self.unroll_length,
            self.batch_size,
            self.num_channels,
            self.frame_dimension,
            self.frame_dimension,
        )
        self.rewards = torch.ones(self.unroll_length, self.batch_size)
        self.done = torch.zeros(self.unroll_length, self.batch_size, dtype=torch.uint8)
        self.episode_return = torch.ones(
            self.unroll_length, self.batch_size
        )  # Not used in the current implemenation of inference.
        self.episode_step = torch.ones(
            self.unroll_length, self.batch_size
        )  # Not used in the current implemenation of inference.

        self.mock_batch = mock.Mock()
        # Set the mock inference batcher to be iterable and return a mock_batch.
        self.mock_inference_batcher = mock.MagicMock()
        self.mock_inference_batcher.__iter__.return_value = iter([self.mock_batch])

    def _test_inference(self, use_lstm, device):
        model = polybeast.Net(num_actions=self.num_actions, use_lstm=use_lstm)
        model.to(device)
        agent_state = model.initial_state()

        inputs = (
            (
                self.frame,
                self.rewards,
                self.done,
                self.episode_return,
                self.episode_return,
            ),
            agent_state,
        )
        # Set the behaviour of the methods of the mock batch.
        self.mock_batch.get_inputs = mock.Mock(return_value=inputs)
        self.mock_batch.set_outputs = mock.Mock()

        # Preparing the mock flags. Could do with just a dict but using
        # a Mock object for consistency.
        mock_flags = mock.Mock()
        mock_flags.actor_device = device
        mock_flags.use_lstm = use_lstm

        polybeast.inference(mock_flags, self.mock_inference_batcher, model)

        # Assert the batch is used only once.
        self.mock_batch.get_inputs.assert_called_once()
        self.mock_batch.set_outputs.assert_called_once()
        # Check that set_outputs has been called with paramaters with the expected shape.
        batch_args, batch_kwargs = self.mock_batch.set_outputs.call_args
        self.assertEqual(batch_kwargs, {})
        model_outputs, *other_args = batch_args
        self.assertEqual(other_args, [])

        (action, policy_logits, baseline), core_state = model_outputs
        self.assertSequenceEqual(action.shape, (self.unroll_length, self.batch_size))
        self.assertSequenceEqual(
            policy_logits.shape, (self.unroll_length, self.batch_size, self.num_actions)
        )
        self.assertSequenceEqual(baseline.shape, (self.unroll_length, self.batch_size))

        for tensor in (action, policy_logits, baseline) + core_state:
            self.assertEqual(tensor.device, torch.device("cpu"))

        self.assertEqual(len(core_state), 2 if use_lstm else 0)
        for core_state_element in core_state:
            self.assertSequenceEqual(
                core_state_element.shape,
                (self.num_lstm_layers, self.batch_size, self.core_output_size),
            )

    def test_inference_cpu_no_lstm(self):
        self._test_inference(use_lstm=False, device=torch.device("cpu"))

    def test_inference_cuda_no_lstm(self):
        if not torch.cuda.is_available():
            warnings.warn("Not testing cuda as it's not available")
            return
        self._test_inference(use_lstm=False, device=torch.device("cuda"))

    def test_inference_cpu_with_lstm(self):
        self._test_inference(use_lstm=True, device=torch.device("cpu"))

    def test_inference_cuda_with_lstm(self):
        if not torch.cuda.is_available():
            warnings.warn("Not testing cuda as it's not available")
            return
        self._test_inference(use_lstm=True, device=torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
