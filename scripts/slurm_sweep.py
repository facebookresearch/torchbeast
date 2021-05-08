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
#
#  Runs a sweep using submitit.
#
#  Test with e.g. python -m scripts.slurm_sweep --local --dry
#
#  Run w/o --local and --dry.
#

import datetime
import getpass
import os
import subprocess
import sys
import itertools

from absl import app
from absl import flags
from absl import logging

import numpy as np

import coolname
import submitit


flags.DEFINE_bool("local", False, "Run locally.")
flags.DEFINE_bool("dry", False, "Don't actually run.")

FLAGS = flags.FLAGS

ENVS = [
    "Adventure",
    "AirRaid",
    "Alien",
    "Amidar",
    "Assault",
    "Asterix",
    "Asteroids",
    "Atlantis",
    "BankHeist",
    "BattleZone",
    "BeamRider",
    "Berzerk",
    "Bowling",
    "Boxing",
    "Breakout",
    "Carnival",
    "Centipede",
    "ChopperCommand",
    "CrazyClimber",
    "Defender",  # gym.make never returns in py3.6.
    "DemonAttack",
    "DoubleDunk",
    "ElevatorAction",
    "FishingDerby",
    "Frostbite",
    "Gopher",
    "Gravitar",
    "Hero",
    "IceHockey",
    "Jamesbond",
    "JourneyEscape",
    "Kangaroo",
    "Krull",
    "KungFuMaster",
    "MontezumaRevenge",
    "MsPacman",
    "NameThisGame",
    "Phoenix",
    "Pitfall",
    "Pong",
    "Pooyan",
    "PrivateEye",
    "Qbert",
    "Riverraid",
    "RoadRunner",
    "Robotank",
    "Seaquest",
    "SpaceInvaders",
    "StarGunner",
    "Tennis",
    "TimePilot",
    "Tutankham",
    "UpNDown",
    "VideoPinball",
    "WizardOfWor",
    "YarsRevenge",
    "Zaxxon",
]

CONFIG = dict(
    env="Pong",
    num_actors=4,
    total_steps=1000,  # int(50e6),
    inference_batch_size=4,
    batch_size=8,
    unroll_length=20,
    use_lstm=False,
    entropy_cost=0.01,
    baseline_cost=0.5,
    discounting=0.99,
    reward_clipping=1.0,
    learning_rate=0.0006,
    alpha=0.99,  # IMPALA paper doesn't even mention this one.
    momentum=0,
    epsilon=0.01,
    grad_norm_clipping=40.0,
    device="cpu",
)

RUNS_PER_CONFIG = 1

SWEEP = dict(env=ENVS[:2], _=range(RUNS_PER_CONFIG))

SIGNED_TO_UNSIGNED_MASK = 2 ** (np.dtype(np.uint).itemsize * 8) - 1

PYTHON_SCRIPT = "scripts.run_single_machine"


def all_configs():
    for vs in itertools.product(*SWEEP.values()):
        config = CONFIG.copy()
        config.update(zip(SWEEP.keys(), vs))
        yield config


def config_to_argv(config):
    yield "python"
    yield "-m"
    yield PYTHON_SCRIPT
    for key, value in config.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bool):
            yield "--" + key if value else "--no" + key
        else:
            yield "--" + key
            yield str(value)


def prefix_string(config):
    result = ""

    keys = (key for key in SWEEP.keys() if not key.startswith("_"))
    keys = ["", ""] + list(sorted(keys))
    for i in range(2, len(keys)):
        length = 1 + max(
            len(os.path.commonprefix(keys[i - 2 : i])),
            len(os.path.commonprefix(keys[i - 1 : i + 1])),
        )
        result += keys[i][:length] + "_" + str(config[keys[i]])
    if "_" in config:
        result += "_" + str(config["_"])
    return result


def log_dir_root():
    if FLAGS.local:
        return os.path.expanduser("~/logs/torchbeast/")
    return "/checkpoint/%s/torchbeast/" % getpass.getuser()


def main(_):
    rootdir = log_dir_root()

    xpid = "%s-%s" % (
        datetime.datetime.now().strftime("%Y%m%d"),
        coolname.generate_slug(),
    )

    xpdir = os.path.join(rootdir, xpid)

    configs = list(all_configs())
    for config in configs:
        log_dir = os.path.join(xpdir, prefix_string(config))
        socketfile = os.path.join(
            "/tmp", "torchbeast-%x" % (hash(log_dir) & SIGNED_TO_UNSIGNED_MASK)
        )

        if not FLAGS.dry:
            os.makedirs(log_dir)
        config["log_dir"] = log_dir
        config["server_address"] = "unix:%s" % socketfile

    settings = dict(
        folder=os.path.join(xpdir, "submitit"), cluster="local" if FLAGS.local else None
    )

    executor = submitit.AutoExecutor(**settings)
    executor.update_parameters(
        partition="learnfair",
        timeout_min=60 * 24 * 2,
        # array_parallelism=2,
        nodes=1,
        tasks_per_node=1,
        # job setup
        name=xpid,
        mem_gb=30,
        cpus_per_task=50,
        gpus_per_node=1,
        constraint="pascal",
    )

    argvs = [list(config_to_argv(config)) for config in configs]
    for argv in argvs:
        print("\n" + " ".join(argv))

    print("\nAbout to submit", len(argvs), "jobs")

    if FLAGS.dry:
        return

    jobs = executor.map_array(subprocess.call, argvs)

    for config, job in zip(configs, jobs):
        print("Submitted with job id:", job.job_id)

        stdout = os.path.join(executor.folder, str(job.job_id) + "_0_log.out")
        stderr = os.path.join(executor.folder, str(job.job_id) + "_0_log.err")

        os.symlink(stdout, os.path.join(config["log_dir"], "submitit.out"))
        os.symlink(stderr, os.path.join(config["log_dir"], "submitit.err"))

        print("stdout ->", stdout)
        print("stderr ->", stderr)

    print("Submitted", len(jobs), "jobs")

    symlink = os.path.join(rootdir, "latest")
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(xpdir, symlink)
        print("Symlinked log directory:", symlink)
    except OSError:
        pass


if __name__ == "__main__":
    app.run(main)
