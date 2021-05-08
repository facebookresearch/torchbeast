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

import timeit
import threading
import concurrent.futures

from absl import app
from absl import flags
from absl import logging

from atari import actor
from atari import learner
from atari import models

import torch
from torch import nn
from torch.nn import functional as F

from torchbeast import queue

import numpy as np

import torch
import nest

FLAGS = flags.FLAGS

OBSERVATION_SHAPE = [4, 84, 84]


def main(_):
    num_actions = 6

    model = models.get_model(num_actions)
    model = model.to(device=FLAGS.device)

    dummy_env_output = (
        torch.empty([1] + OBSERVATION_SHAPE, dtype=torch.uint8),
        torch.zeros(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.bool),
    )

    T = FLAGS.unroll_length
    B = FLAGS.batch_size

    dummy_model_input = dict(
        last_actions=torch.zeros([1], dtype=torch.int64),
        env_outputs=dummy_env_output,
        core_state=(),
    )
    dummy_model_input = nest.map(lambda t: t.to(FLAGS.device), dummy_model_input)

    dummy_model_output, _ = model(**dummy_model_input)

    batch = dict(
        last_actions=dummy_model_input["last_actions"],
        env_outputs=dummy_model_input["env_outputs"],
        actor_outputs=dummy_model_output,
    )

    def expand(t):
        shape = [1] + list(t.shape)
        shape[1] = T
        return t.cpu().expand(shape).contiguous()

    batch = nest.map(expand, batch)
    initial_agent_state = ()

    dummy_model_input = nest.map(lambda t: t.cpu(), dummy_model_input)

    unroll_queue = queue.Queue(maxsize=1)
    log_queue = queue.Queue()

    def enqueue():
        try:
            while True:
                data = nest.map(
                    lambda t: t.clone(), (torch.arange(1), batch, initial_agent_state)
                )
                # data = nest.map(lambda t: t.cuda(), data)
                # unroll_queue.put(nest.map(lambda t: t.cpu(), data), timeout=5.0)
                unroll_queue.put(data, timeout=5.0)
        except queue.Closed:
            return

    NUM_ENQUEUES = 2

    threads = []
    for _ in range(NUM_ENQUEUES):
        t = threading.Thread(target=enqueue)
        t.start()
        threads.append(t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            learner.learn(model, executor, unroll_queue, log_queue)
        except KeyboardInterrupt:
            print("Stopping ...")
        finally:
            unroll_queue.close()
            for t in threads:
                t.join()


if __name__ == "__main__":
    app.run(main)
