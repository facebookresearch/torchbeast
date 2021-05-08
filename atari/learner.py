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

import concurrent.futures
import csv
import json
import os
import time
import timeit

# Necessary for multithreading.
os.environ["OMP_NUM_THREADS"] = "1"  # noqa: E402

from absl import app
from absl import flags
from absl import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import nest

from atari import environment
from atari import models
from atari import vtrace

import torchbeast
from torchbeast import queue


FLAGS = flags.FLAGS

# Training.
flags.DEFINE_integer("num_actors", 4, "Number of actors.")
flags.DEFINE_integer("batch_size", 8, "Batch size for learning.")
flags.DEFINE_integer("unroll_length", 20, "Unroll length.")
flags.DEFINE_integer("total_steps", 30000, "Total number of steps to train for.")
flags.DEFINE_integer("inference_batch_size", 2, "Batch size for inference.")

# Loss settings.
flags.DEFINE_float("entropy_cost", 0.0006, "Entropy cost/multiplier.")
flags.DEFINE_float("baseline_cost", 0.5, "Baseline cost/multiplier.")
flags.DEFINE_float("discounting", 0.99, "Discounting factor.")
flags.DEFINE_float(
    "reward_clipping", 1.0, "Maximum absolute reward. Set to 0.0 to disable."
)

# Optimizer settings.
flags.DEFINE_float("learning_rate", 0.00048, "Learning rate.")
flags.DEFINE_float("alpha", 0.99, "RMSProp smoothing constant.")
flags.DEFINE_float("momentum", 0, "RMSProp momentum.")
flags.DEFINE_float("epsilon", 0.01, "RMSProp epsilon.")
flags.DEFINE_float(
    "grad_norm_clipping", 40.0, "Global gradient norm clip. Set to 0.0 to disable."
)

# Hardware settings
flags.DEFINE_string("inference_device", "cuda:0", "Device for inference.")
flags.DEFINE_string("learner_device", "cuda:1", "Device for learning.")


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def compute_loss(learner_outputs, actor_outputs, env_outputs, last_actions):
    del last_actions  # Only used in model.
    bootstrap_value = learner_outputs["baseline"][-1]

    # Move from obs[t] -> action[t] to action[t] -> obs[t].
    # TODO: Check this logic! It does not match Polybeast ?!
    actor_outputs = nest.map(lambda t: t[:-1], actor_outputs)
    rewards, done = nest.map(lambda t: t[1:], env_outputs[1:])
    learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

    if FLAGS.reward_clipping:
        rewards = torch.clamp(rewards, -FLAGS.reward_clipping, FLAGS.reward_clipping)

    rewards = rewards.float()
    discounts = (~done).float() * FLAGS.discounting

    actor_logits = actor_outputs["policy_logits"]
    learner_logits = learner_outputs["policy_logits"]
    actions = actor_outputs["action"]
    baseline = learner_outputs["baseline"]

    vtrace_returns = vtrace.from_logits(
        behavior_policy_logits=actor_logits,
        target_policy_logits=learner_logits,
        actions=actions,
        discounts=discounts,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
    )

    pg_loss = compute_policy_gradient_loss(
        learner_logits, actions, vtrace_returns.pg_advantages
    )
    baseline_loss = FLAGS.baseline_cost * compute_baseline_loss(
        vtrace_returns.vs - baseline
    )
    entropy_loss = FLAGS.entropy_cost * compute_entropy_loss(learner_logits)

    return pg_loss + baseline_loss + entropy_loss


class Rollouts:
    def __init__(self, timestep, num_overlapping_steps=0):
        self._full_length = num_overlapping_steps + FLAGS.unroll_length + 1
        self._num_overlapping_steps = num_overlapping_steps

        N = FLAGS.num_actors
        L = self._full_length

        self._state = nest.map(
            lambda t: torch.zeros((N, L) + t.shape, dtype=t.dtype), timestep
        )

        self._index = torch.zeros([N], dtype=torch.int64)

    def append(self, actor_ids, timesteps):
        assert len(actor_ids) == len(actor_ids.unique()), "Duplicate actor ids"
        for s in nest.flatten(timesteps):
            assert s.shape[0] == actor_ids.shape[0], "Batch dimension don't match"

        curr_indices = self._index[actor_ids]

        for s, v in zip(nest.flatten(self._state), nest.flatten(timesteps)):
            s[actor_ids, curr_indices] = v

        self._index[actor_ids] += 1

        return self._complete_unrolls(actor_ids)

    def _complete_unrolls(self, actor_ids):
        actor_indices = self._index[actor_ids]

        actor_ids = actor_ids[actor_indices == self._full_length]
        unrolls = nest.map(lambda s: s[actor_ids], self._state)

        j = self._num_overlapping_steps + 1
        for s in nest.flatten(self._state):
            s[actor_ids, :j] = s[actor_ids, -j:]

        self._index.scatter_(0, actor_ids, 1 + self._num_overlapping_steps)

        return actor_ids, unrolls

    def reset(self, actor_ids):
        j = self._num_overlapping_steps
        self._index.scatter_(0, actor_ids, j)

        for s in nest.flatten(self._state):
            # .zero_() doesn't work with tensor indexing?
            s[actor_ids, :j] = 0


class StructuredBuffer:
    def __init__(self, state):
        self._state = state

    def get(self, ids):
        return nest.map(lambda s: s[ids], self._state)

    def set(self, ids, values):
        for s, v in zip(nest.flatten(self._state), nest.flatten(values)):
            s[ids] = v

    def add(self, ids, values):
        for s, v in zip(nest.flatten(self._state), nest.flatten(values)):
            s[ids] += v

    def clear(self, ids):
        for s in nest.flatten(self._state):
            # .zero_() doesn't work with tensor indexing?
            s[ids] = 0

    def __str__(self):
        return str(self._state)


def checkpoint(model, optimizer, scheduler):
    if not FLAGS.log_dir:
        return

    path = os.path.join(FLAGS.log_dir, "checkpoint.tar")
    logging.info("Saving checkpoint to %s", path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "flags": FLAGS.flag_values_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, scheduler):
    if not FLAGS.log_dir:
        return

    path = os.path.join(FLAGS.log_dir, "checkpoint.tar")
    if not os.path.exists(path):
        return

    state_dict = torch.load(path)
    model.load_state_dict(state_dict["model_state_dict"])
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
    scheduler.load_state_dict(state_dict["scheduler_state_dict"])

    logging.info("Loaded checkpoint from %s", path)

    flags_dict = FLAGS.flag_values_dict()
    flags_diff = state_dict["flags"].items() ^ flags_dict.items()
    if flags_diff:
        logging.warn(
            "Flags have changed since checkpoint was written. Difference: %s",
            flags_diff,
        )


def log(_state={}, **fields):  # noqa: B008
    if "writer" not in _state:
        if not FLAGS.log_dir:
            _state["writer"] = None
            return

        path = os.path.join(FLAGS.log_dir, "logs.tsv")
        writeheader = not os.path.exists(path)
        fieldnames = list(fields.keys())

        _state["file"] = open(path, "a", buffering=1)  # Line buffering.
        _state["writer"] = csv.DictWriter(_state["file"], fieldnames, delimiter="\t")
        if writeheader:
            _state["writer"].writeheader()

    writer = _state["writer"]
    if writer is not None:
        writer.writerow(fields)


def learner_loop():
    if FLAGS.num_actors < FLAGS.batch_size:
        logging.warn("Batch size is larger than number of actors.")
    assert (
        FLAGS.batch_size % FLAGS.inference_batch_size == 0
    ), "For now, inference_batch_size must divide batch_size"
    if FLAGS.log_dir:
        log_file_path = os.path.join(FLAGS.log_dir, "logs.tsv")
        logging.info(
            "%s logs to %s",
            "Appending" if os.path.exists(log_file_path) else "Writing",
            log_file_path,
        )
    else:
        logging.warn("--log_dir not set. Not writing logs to file.")

    unroll_queue = queue.Queue(maxsize=1)
    log_queue = queue.Queue()

    env = environment.create_env()
    env.reset()

    num_actions = env.action_space.n

    model = models.get_model(num_actions)

    dummy_env_output = env.step(0)[:-1]  # Drop info.
    dummy_env_output = nest.map(
        lambda a: torch.from_numpy(np.array(a)), dummy_env_output
    )

    with torch.no_grad():
        dummy_model_output, _ = model(
            last_actions=torch.zeros([1], dtype=torch.int64),
            env_outputs=nest.map(lambda t: t.unsqueeze(0), dummy_env_output),
            core_state=model.initial_state(1),
        )
        dummy_model_output = nest.map(lambda t: t.squeeze(0), dummy_model_output)

    model = model.to(device=FLAGS.inference_device)

    # TODO: Decide if we really want that for simple tensors?
    actions = StructuredBuffer(torch.zeros([FLAGS.num_actors], dtype=torch.int64))
    actor_run_ids = StructuredBuffer(torch.zeros([FLAGS.num_actors], dtype=torch.int64))
    actor_infos = StructuredBuffer(
        dict(
            episode_step=torch.zeros([FLAGS.num_actors], dtype=torch.int64),
            episode_return=torch.zeros([FLAGS.num_actors]),
        )
    )

    # Agent states at the beginning of an unroll. Needs to be kept for learner.
    first_agent_states = StructuredBuffer(
        model.initial_state(batch_size=FLAGS.num_actors)
    )

    # Current agent states.
    agent_states = StructuredBuffer(model.initial_state(batch_size=FLAGS.num_actors))

    rollouts = Rollouts(
        dict(
            last_actions=torch.zeros((), dtype=torch.int64),
            env_outputs=dummy_env_output,
            actor_outputs=dummy_model_output,
        )
    )

    server = torchbeast.Server(FLAGS.server_address, max_parallel_calls=4)

    def inference(actor_ids, run_ids, env_outputs):
        torch.set_grad_enabled(False)
        previous_run_ids = actor_run_ids.get(actor_ids)
        reset_indices = previous_run_ids != run_ids
        actor_run_ids.set(actor_ids, run_ids)

        actors_needing_reset = actor_ids[reset_indices]

        # Update new/restarted actors.
        if actors_needing_reset.numel():
            logging.info("Actor ids needing reset: %s", actors_needing_reset.tolist())

            actor_infos.clear(actors_needing_reset)
            rollouts.reset(actors_needing_reset)
            actions.clear(actors_needing_reset)

            initial_agent_states = model.initial_state(actors_needing_reset.numel())
            first_agent_states.set(actors_needing_reset, initial_agent_states)
            agent_states.set(actors_needing_reset, initial_agent_states)

        observation, reward, done = env_outputs

        # Update logging stats.
        actor_infos.add(actor_ids, dict(episode_step=0, episode_return=reward))
        done_ids = actor_ids[done]
        if done_ids.numel():
            log_queue.put((done_ids, actor_infos.get(done_ids)))
            actor_infos.clear(done_ids)
        actor_infos.add(actor_ids, dict(episode_step=1, episode_return=0.0))

        last_actions = actions.get(actor_ids)
        prev_agent_states = agent_states.get(actor_ids)

        actor_outputs, new_agent_states = model(
            *nest.map(
                lambda t: t.to(FLAGS.inference_device),
                (last_actions, env_outputs, prev_agent_states),
            )
        )
        actor_outputs, new_agent_states = nest.map(
            lambda t: t.cpu(), (actor_outputs, new_agent_states)
        )

        timestep = dict(
            last_actions=last_actions,
            env_outputs=env_outputs,
            actor_outputs=actor_outputs,
        )
        completed_ids, unrolls = rollouts.append(actor_ids, timestep)
        if completed_ids.numel():
            try:
                unroll_queue.put(
                    (completed_ids, unrolls, first_agent_states.get(completed_ids)),
                    timeout=5.0,
                )
            except queue.Closed:
                if server.running():
                    raise

        first_agent_states.set(completed_ids, agent_states.get(completed_ids))
        agent_states.set(actor_ids, new_agent_states)

        action = actor_outputs["action"]
        actions.set(actor_ids, action)
        return action

    server.bind("inference", inference, batch_size=FLAGS.inference_batch_size)
    server.run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        try:
            learn(model, executor, unroll_queue, log_queue)
        except KeyboardInterrupt:
            print("Stopping ...")
        finally:
            unroll_queue.close()
            server.stop()
        # Need to shut down executor after queue is closed.


def learn(inference_model, executor, unroll_queue, log_queue):
    model = models.get_model(inference_model.num_actions)
    model.to(FLAGS.learner_device)

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
    load_checkpoint(model, optimizer, scheduler)
    inference_model.load_state_dict(model.state_dict())

    steps_per_epoch = FLAGS.batch_size * FLAGS.unroll_length
    steps = scheduler.last_epoch * steps_per_epoch

    last_step = steps
    last_time = timeit.default_timer()

    current_time = last_time
    last_checkpoint_time = last_time

    def load_on_gpu():
        # TODO: Use CUDA streams?
        entries = 0
        next_batch = []

        while entries < FLAGS.batch_size:
            # TODO: This isn't guaranteed to be exact if inference_batch_size
            # does not divide batch_size evenly.
            ids, *data = unroll_queue.get()
            next_batch.append(data)
            entries += ids.numel()

        # Batch.
        batch, initial_agent_state = nest.map_many(lambda d: torch.cat(d), *next_batch)

        # Make time major (excluding agent states).
        for t in nest.flatten(batch):
            t.transpose_(0, 1)

        if not FLAGS.learner_device.startswith("cuda"):
            return nest.map(lambda t: t.contiguous(), (batch, initial_agent_state))
        return nest.map(
            lambda t: t.to(FLAGS.learner_device), (batch, initial_agent_state)
        )

    def log_target(steps, current_time, loss_value):
        nonlocal last_step
        nonlocal last_time

        sps = (steps - last_step) / (current_time - last_time)
        last_step = steps
        last_time = current_time

        episode_returns = []

        for _ in range(log_queue.qsize()):
            ids, infos = log_queue.get()
            for actor_id, episode_step, episode_return in zip(
                ids.tolist(),
                infos["episode_step"].tolist(),
                infos["episode_return"].tolist(),
            ):
                episode_returns.append(episode_return)

                log(
                    step=steps,
                    episode_step=episode_step,
                    episode_return=episode_return,
                    actor_id=actor_id,
                    sps=sps,
                    loss=loss_value,
                    timestep=time.time(),
                )

        if steps % 100 == 0:
            if episode_returns:
                logging.info(
                    "Step %i @ %.1f SPS. Mean episode return: %f. "
                    "Episodes finished: %i. Loss: %f.",
                    steps,
                    sps,
                    sum(episode_returns) / len(episode_returns),
                    len(episode_returns),
                    loss_value,
                )
            else:
                logging.info("Step %i @ %.1f SPS. Loss: %f.", steps, sps, loss_value)

    batch_future = executor.submit(load_on_gpu)
    log_future = executor.submit(lambda: None)

    while steps < FLAGS.total_steps:
        batch, initial_agent_state = batch_future.result()
        batch_future = executor.submit(load_on_gpu)

        learner_outputs, _ = model(
            batch["last_actions"],
            batch["env_outputs"],
            initial_agent_state,
            unroll=True,
        )

        loss = compute_loss(learner_outputs, **batch)

        optimizer.zero_grad()
        loss.backward()
        if FLAGS.grad_norm_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), FLAGS.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        steps += steps_per_epoch
        loss_value = loss.item()

        current_time = timeit.default_timer()

        if current_time - last_checkpoint_time > 10 * 60:  # Every 10 min.
            checkpoint(model, optimizer, scheduler)
            last_checkpoint_time = timeit.default_timer()

        inference_model.load_state_dict(model.state_dict())

        log_future.result()
        log_future = executor.submit(log_target, steps, current_time, loss_value)

    logging.info("Learning finished after %i steps", steps)
    checkpoint(model, optimizer, scheduler)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if FLAGS.log_dir:
        logging.get_absl_handler().use_absl_log_file(program_name=FLAGS.role)

        # Write meta.json file with some information on our setup.
        metadata = {
            "flags": FLAGS.flag_values_dict(),
            "env": os.environ.copy(),
            "date_start": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            import git
        except ImportError:
            pass
        else:
            try:
                repo = git.Repo(search_parent_directories=True)
                metadata["git"] = {
                    "commit": repo.commit().hexsha,
                    "is_dirty": repo.is_dirty(),
                    "path": repo.git_dir,
                }
                if not repo.head.is_detached:
                    metadata["git"]["branch"] = repo.active_branch.name
            except git.InvalidGitRepositoryError:
                pass

        if "git" not in metadata:
            logging.warn("Couldn't determine git data.")

        with open(os.path.join(FLAGS.log_dir, "meta.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    logging.info(
        "Starting %s with FLAGS: \n%s",
        FLAGS.role,
        json.dumps(FLAGS.flag_values_dict(), indent=2, sort_keys=True),
    )
    learner_loop()


if __name__ == "__main__":
    app.run(main)
