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

import random

import numpy as np

import torchbeast


client_id = random.randint(0, 10000)

print("Client with id", client_id)


client = torchbeast.Client("localhost:12345")

client.connect(3)

print(client.myfunc((np.zeros((1, 2), dtype=np.float32))))
print(client.pyfunc(np.zeros((1, 2), dtype=np.float32)))


client_array = np.array([0, client_id, 2 * client_id])

inputs = (0, 1, (client_array, np.array(True)))

client.identity(inputs)

np.testing.assert_array_equal(client_array, client.identity(client_array))
np.testing.assert_array_equal(client_array, client.batched_identity(client_array))
