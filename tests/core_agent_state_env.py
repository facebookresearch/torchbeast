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
"""Mock environment for the test core_agent_state_test.py."""

import numpy as np

import libtorchbeast


class Env:
    def __init__(self):
        self.frame = np.zeros((1, 1))
        self.count = 0
        self.done_after = 5

    def reset(self):
        self.frame = np.zeros((1, 1))
        return self.frame

    def step(self, action):
        self.frame += 1
        done = self.frame.item() == self.done_after
        return self.frame, 0.0, done, {}


if __name__ == "__main__":
    server_address = "unix:/tmp/core_agent_state_test"
    server = libtorchbeast.Server(Env, server_address=server_address)
    server.run()
