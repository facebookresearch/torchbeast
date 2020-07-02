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
"""Tests for polybeast Net class implementation."""

import unittest

import torch
from torchbeast import polybeast_learner as polybeast


class NetTest(unittest.TestCase):
    def setUp(self):
        self.unroll_length = 4  # Arbitrary.
        self.batch_size = 4  # Arbitrary.
        self.frame_dimension = 84  # Has to match what expected by the model.
        self.num_actions = 6  # Specific to each environment.
        self.num_channels = 4  # Has to match with the first conv layer of the net.
        self.core_output_size = 256  # Has to match what expected by the model.
        self.num_lstm_layers = 1  # As in the model.

        self.inputs = dict(
            frame=torch.ones(
                self.unroll_length,
                self.batch_size,
                self.num_channels,
                self.frame_dimension,
                self.frame_dimension,
            ),
            reward=torch.ones(self.batch_size, self.unroll_length),
            done=torch.zeros(self.batch_size, self.unroll_length, dtype=torch.uint8),
        )

    def test_forward_return_signature_no_lstm(self):
        model = polybeast.Net(num_actions=self.num_actions, use_lstm=False)
        core_state = ()

        (action, policy_logits, baseline), core_state = model(self.inputs, core_state)
        self.assertSequenceEqual(action.shape, (self.batch_size, self.unroll_length))
        self.assertSequenceEqual(
            policy_logits.shape, (self.batch_size, self.unroll_length, self.num_actions)
        )
        self.assertSequenceEqual(baseline.shape, (self.batch_size, self.unroll_length))
        self.assertSequenceEqual(core_state, ())

    def test_forward_return_signature_with_lstm(self):
        model = polybeast.Net(num_actions=self.num_actions, use_lstm=True)
        core_state = model.initial_state(self.batch_size)

        (action, policy_logits, baseline), core_state = model(self.inputs, core_state)
        self.assertSequenceEqual(action.shape, (self.batch_size, self.unroll_length))
        self.assertSequenceEqual(
            policy_logits.shape, (self.batch_size, self.unroll_length, self.num_actions)
        )
        self.assertSequenceEqual(baseline.shape, (self.batch_size, self.unroll_length))
        self.assertEqual(len(core_state), 2)
        for core_state_element in core_state:
            self.assertSequenceEqual(
                core_state_element.shape,
                (self.num_lstm_layers, self.batch_size, self.core_output_size),
            )

    def test_initial_state(self):
        model_no_lstm = polybeast.Net(num_actions=self.num_actions, use_lstm=False)
        initial_state_no_lstm = model_no_lstm.initial_state(self.batch_size)
        self.assertSequenceEqual(initial_state_no_lstm, ())

        model_with_lstm = polybeast.Net(num_actions=self.num_actions, use_lstm=True)
        initial_state_with_lstm = model_with_lstm.initial_state(self.batch_size)
        self.assertEqual(len(initial_state_with_lstm), 2)
        for core_state_element in initial_state_with_lstm:
            self.assertSequenceEqual(
                core_state_element.shape,
                (self.num_lstm_layers, self.batch_size, self.core_output_size),
            )


if __name__ == "__main__":
    unittest.main()
