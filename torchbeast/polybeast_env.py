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
import threading
import time

import numpy as np
import libtorchbeast
from torchbeast import atari_wrappers


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument('--num_servers', default=4, type=int, metavar='N',
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Gym environment.')
# yapf: enable


class Env:
    def reset(self):
        print("reset called")
        return np.ones((4, 84, 84), dtype=np.uint8)

    def step(self, action):
        frame = np.zeros((4, 84, 84), dtype=np.uint8)
        return frame, 0.0, False, {}  # First three mandatory.


def create_env(env_name, lock=threading.Lock()):
    with lock:  # Atari isn't threadsafe at construction time.
        return atari_wrappers.wrap_pytorch(
            atari_wrappers.wrap_deepmind(
                atari_wrappers.make_atari(env_name),
                clip_rewards=False,
                frame_stack=True,
                scale=False,
            )
        )


def serve(env_name, server_address):
    init = Env if env_name == "Mock" else lambda: create_env(env_name)
    server = libtorchbeast.Server(init, server_address=server_address)
    server.run()


def main(flags):
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    processes = []
    for i in range(flags.num_servers):
        p = mp.Process(
            target=serve, args=(flags.env, f"{flags.pipes_basename}.{i}"), daemon=True
        )
        p.start()
        processes.append(p)

    try:
        # We are only here to listen to the interrupt.
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
