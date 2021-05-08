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

import collections
import unittest
import multiprocessing as mp

import numpy as np

import torch

import torchbeast


def _run_client():
    client = torchbeast.Client("localhost:12345")
    client.connect(10)
    client.first_function(
        np.zeros((1, 2)), np.arange(10), (np.random.uniform((2, 3)), np.ones((1, 2)))
    )
    # client.batched_function(np.zeros((1, 2)))


class RPCTest(unittest.TestCase):
    def test_rpc_simple(self):
        np.random.seed(0)
        client_process = mp.Process(target=_run_client)

        calls = collections.defaultdict(int)

        def first_function(a, b, c):
            calls["first_function"] += 1
            np.testing.assert_array_equal(a.numpy(), np.zeros((1, 2)))
            np.testing.assert_array_equal(b.numpy(), np.arange(10))

            c0, c1 = c
            np.testing.assert_array_equal(c0.numpy(), np.random.uniform((2, 3)))
            np.testing.assert_array_equal(c1.numpy(), np.ones((1, 2)))

            return torch.ones(1, 1)

        server = torchbeast.Server("localhost:12345")
        server.bind("first_function", first_function)
        server.run()

        client_process.start()

        client_process.join()
        server.stop()

        self.assertEqual(calls["first_function"], 1)

    # TODO(heiner): Add more tests. Batching, return values, etc.


if __name__ == "__main__":
    unittest.main()
