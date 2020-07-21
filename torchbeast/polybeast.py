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

import argparse
import multiprocessing as mp

import numpy as np

from torchbeast import polybeast_learner
from torchbeast import polybeast_env


def run_env(flags, actor_id):
    np.random.seed()  # Get new random seed in forked process.
    polybeast_env.main(flags)


def run_learner(flags):
    polybeast_learner.main(flags)


def main():
    flags = argparse.Namespace()
    flags, argv = polybeast_learner.parser.parse_known_args(namespace=flags)
    flags, argv = polybeast_env.parser.parse_known_args(args=argv, namespace=flags)
    if argv:
        # Produce an error message.
        polybeast_learner.parser.print_usage()
        print("")
        polybeast_env.parser.print_usage()
        print("Unkown args:", " ".join(argv))
        return -1

    env_processes = []
    for actor_id in range(1):
        p = mp.Process(target=run_env, args=(flags, actor_id))
        p.start()
        env_processes.append(p)

    run_learner(flags)

    for p in env_processes:
        p.join()


if __name__ == "__main__":
    main()
