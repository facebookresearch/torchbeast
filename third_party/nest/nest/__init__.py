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

from ._nest import *


def flatten(n):
    if isinstance(n, tuple) or isinstance(n, list):
        for sn in n:
            yield from flatten(sn)
    elif isinstance(n, dict):
        for key in sorted(n.keys()):  # C++ nests are std::maps ordered by key.
            yield from flatten(n[key])
    else:
        yield n
