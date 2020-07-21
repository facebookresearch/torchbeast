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
"""Test that non-contiguous arrays are handled properly."""

import subprocess
import threading
import unittest

import numpy as np

import torch

import libtorchbeast


class ContiguousArraysTest(unittest.TestCase):
    def setUp(self):
        self.server_proc = subprocess.Popen(
            ["python", "tests/contiguous_arrays_env.py"]
        )

        server_address = ["unix:/tmp/contiguous_arrays_test"]
        self.learner_queue = libtorchbeast.BatchingQueue(
            batch_dim=1, minimum_batch_size=1, maximum_batch_size=10, check_inputs=True
        )
        self.inference_batcher = libtorchbeast.DynamicBatcher(
            batch_dim=1,
            minimum_batch_size=1,
            maximum_batch_size=10,
            timeout_ms=100,
            check_outputs=True,
        )
        actor = libtorchbeast.ActorPool(
            unroll_length=1,
            learner_queue=self.learner_queue,
            inference_batcher=self.inference_batcher,
            env_server_addresses=server_address,
            initial_agent_state=(),
        )

        def run():
            actor.run()

        self.actor_thread = threading.Thread(target=run)
        self.actor_thread.start()

        self.target = np.arange(3 * 4 * 5)
        self.target = self.target.reshape(3, 4, 5)
        self.target = self.target.transpose(2, 1, 0)

    def check_inference_inputs(self):
        batch = next(self.inference_batcher)
        batched_env_outputs, _ = batch.get_inputs()
        frame, *_ = batched_env_outputs
        self.assertTrue(np.array_equal(frame.shape, (1, 1, 5, 4, 3)))
        frame = frame.reshape(5, 4, 3)
        self.assertTrue(np.array_equal(frame, self.target))
        # Set an arbitrary output.
        batch.set_outputs(((torch.ones(1, 1),), ()))

    def test_contiguous_arrays(self):
        self.check_inference_inputs()
        # Stop actor thread.
        self.inference_batcher.close()
        self.learner_queue.close()
        self.actor_thread.join()

    def tearDown(self):
        self.server_proc.terminate()
