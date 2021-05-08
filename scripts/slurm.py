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
#  Runs a single experiment using submitit.
#
#  Test with e.g. python -m scripts.slurm --local -- --total_steps=10000 --device=cpu
#
#  Run w/o the --local
#

import datetime
import getpass
import os
import subprocess
import sys

from absl import app
from absl import flags
from absl import logging

import numpy as np

import coolname
import submitit


flags.DEFINE_bool("local", False, "Run locally.")
flags.DEFINE_integer("num_actors", 4, "Number of actors.")

FLAGS = flags.FLAGS


SIGNED_TO_UNSIGNED_MASK = 2 ** (np.dtype(np.uint).itemsize * 8) - 1


def run(executor, config):
    args = [
        "python",
        "-m",
        "scripts.run_single_machine",
        "--server_address=unix:%s" % config["socketfile"],
        "--num_actors=%i" % FLAGS.num_actors,
        "--log_dir=%s" % config["folder"],
    ]
    if "--" in sys.argv:
        args.extend(sys.argv[sys.argv.index("--") + 1 :])
    return executor.submit(subprocess.call, args)


def log_dir_root():
    if FLAGS.local:
        return os.path.expanduser("~/logs/torchbeast/")
    return "/checkpoint/%s/torchbeast/" % getpass.getuser()


def main(argv):
    rootdir = log_dir_root()
    os.makedirs(rootdir, exist_ok=True)

    xpid = "%s-%s" % (
        datetime.datetime.now().strftime("%Y%m%d"),
        coolname.generate_slug(),
    )

    config = dict(
        folder=os.path.join(rootdir, xpid), cluster="local" if FLAGS.local else None
    )

    executor = submitit.AutoExecutor(**config)
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

    xpidhash = hash(xpid) & SIGNED_TO_UNSIGNED_MASK
    config.update(dict(socketfile=os.path.join("/tmp", "torchbeast-%x" % xpidhash)))

    jobs = [run(executor, config)]

    for job in jobs:
        print("Submitted with job id: ", job.job_id)
        print(f"stdout -> {executor.folder}/{job.job_id}_0_log.out")
        print(f"stderr -> {executor.folder}/{job.job_id}_0_log.err")

    symlink = os.path.join(rootdir, "latest")
    try:
        if os.path.islink(symlink):
            os.remove(symlink)
        if not os.path.exists(symlink):
            os.symlink(executor.folder, symlink)
        print("Symlinked log directory:", symlink)
    except OSError:
        pass


if __name__ == "__main__":
    app.run(main)
