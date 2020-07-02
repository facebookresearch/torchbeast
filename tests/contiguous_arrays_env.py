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
"""Mock environment for the test contiguous_arrays_test.py."""

import numpy as np
import libtorchbeast


class Env:
    def __init__(self):
        self.frame = np.arange(3 * 4 * 5)
        self.frame = self.frame.reshape(3, 4, 5)
        self.frame = self.frame.transpose(2, 1, 0)
        assert not self.frame.flags.c_contiguous

    def reset(self):
        return self.frame

    def step(self, action):
        return self.frame, 0.0, False, {}


if __name__ == "__main__":
    server_address = "unix:/tmp/contiguous_arrays_test"
    server = libtorchbeast.Server(Env, server_address=server_address)
    server.run()
