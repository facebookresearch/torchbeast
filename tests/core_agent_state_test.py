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
"""Test that the core state is handled correctly by the batching mechanism."""

import unittest
import threading
import subprocess

import torch
from torch import nn

import libtorchbeast


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def initial_state(self):
        return torch.zeros(1, 1)

    def forward(self, inputs, core_state):
        x = inputs["frame"]
        notdone = (~inputs["done"]).float()
        T, B, *_ = x.shape

        for nd in notdone.unbind():
            nd.view(1, -1)
            core_state = nd * core_state
            core_state = core_state + 1
        # Arbitrarily return action 1.
        action = torch.ones((T, B), dtype=torch.int32)
        return (action,), core_state


class CoreAgentStateTest(unittest.TestCase):
    def setUp(self):
        self.server_proc = subprocess.Popen(["python", "tests/core_agent_state_env.py"])

        self.B = 2
        self.T = 3
        self.model = Net()
        server_address = ["unix:/tmp/core_agent_state_test"]
        self.learner_queue = libtorchbeast.BatchingQueue(
            batch_dim=1,
            minimum_batch_size=self.B,
            maximum_batch_size=self.B,
            check_inputs=True,
        )
        self.inference_batcher = libtorchbeast.DynamicBatcher(
            batch_dim=1,
            minimum_batch_size=1,
            maximum_batch_size=1,
            timeout_ms=100,
            check_outputs=True,
        )
        self.actor = libtorchbeast.ActorPool(
            unroll_length=self.T,
            learner_queue=self.learner_queue,
            inference_batcher=self.inference_batcher,
            env_server_addresses=server_address,
            initial_agent_state=self.model.initial_state(),
        )

    def inference(self):
        for batch in self.inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            frame, _, done, *_ = batched_env_outputs
            # Check that when done is set we reset the environment.
            # Since we only have one actor producing experience we will always
            # have batch_size == 1, hence we can safely use item().
            if done.item():
                self.assertEqual(frame.item(), 0.0)
            outputs = self.model(dict(frame=frame, done=done), agent_state)
            batch.set_outputs(outputs)

    def learn(self):
        for i, tensors in enumerate(self.learner_queue):
            batch, initial_agent_state = tensors
            env_outputs, actor_outputs = batch
            frame, _, done, *_ = env_outputs
            # Make sure the last env_outputs of a rollout equals the first of the
            # following one.
            # This is guaranteed to be true if there is only one actor filling up
            # the learner queue.
            self.assertEqual(frame[self.T][0].item(), frame[0][1].item())
            self.assertEqual(done[self.T][0].item(), done[0][1].item())

            # Make sure the initial state equals the value of the frame at the beginning
            # of the rollout. This has to be the case in our test since:
            # - every call to forward increments the core state by one.
            # - every call to step increments the value in the frame by one (modulo 5).
            env_done_after = 5  # Matches self.done_after in core_agent_state_env.py.
            self.assertEqual(
                frame[0][0].item(), initial_agent_state[0][0].item() % env_done_after
            )
            self.assertEqual(
                frame[0][1].item(), initial_agent_state[0][1].item() % env_done_after
            )

            if i >= 10:
                # Stop execution.
                self.learner_queue.close()
                self.inference_batcher.close()

    def test_core_agent_state(self):
        def run():
            self.actor.run()

        threads = [
            threading.Thread(target=self.inference),
            threading.Thread(target=run),
        ]

        # Start actor and inference thread.
        for thread in threads:
            thread.start()

        self.learn()

        for thread in threads:
            thread.join()

    def tearDown(self):
        self.server_proc.terminate()
        self.server_proc.wait()


if __name__ == "__main__":
    unittest.main()
