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

from absl import app
from absl import flags
from absl import logging

from atari import actor
from atari import learner
from atari import models

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
        core_state=model.initial_state(1),
    )
    dummy_model_input = nest.map(lambda t: t.to(FLAGS.device), dummy_model_input)

    dummy_model_output, _ = model(**dummy_model_input)

    batch = dict(
        last_actions=dummy_model_input["last_actions"],
        env_outputs=dummy_model_input["env_outputs"],
        actor_outputs=dummy_model_output,
    )

    initial_agent_state = nest.map(lambda t: t.to(FLAGS.device), model.initial_state(B))

    def expand(t):
        shape = [T] + list(t.shape)
        shape[1] = B
        return t.expand(shape).contiguous().to(device=FLAGS.device)

    batch = nest.map(expand, batch)

    last_time = timeit.default_timer()

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=FLAGS.learning_rate,
        momentum=FLAGS.momentum,
        eps=FLAGS.epsilon,
        alpha=FLAGS.alpha,
    )

    def lr_lambda(epoch):
        return (
            1
            - min(epoch * FLAGS.unroll_length * FLAGS.batch_size, FLAGS.total_steps)
            / FLAGS.total_steps
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    steps = 0
    steps_per_epoch = FLAGS.batch_size * FLAGS.unroll_length
    last_step = 0

    for _ in range(6):
        learner_outputs, _ = model(
            batch["last_actions"],
            batch["env_outputs"],
            initial_agent_state,
            unroll=True,
        )
        loss = learner.compute_loss(learner_outputs, **batch)

        optimizer.zero_grad()
        loss.backward()
        if FLAGS.grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()

        steps += steps_per_epoch

        current_time = timeit.default_timer()

        sps = (steps - last_step) / (current_time - last_time)
        last_step = steps
        last_time = current_time

        logging.info("Step %i @ %.1f SPS. Loss: %f.", steps, sps, loss_value)


if __name__ == "__main__":
    app.run(main)
