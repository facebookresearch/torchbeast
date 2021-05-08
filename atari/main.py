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

from absl import app
from absl import flags

from atari import learner
from atari import actor

FLAGS = flags.FLAGS

flags.DEFINE_enum("role", None, ["learner", "actor"], "Role to run at.")
flags.DEFINE_string(
    "server_address",
    "localhost:12345",
    "Address of the server (e.g. localhost:12345 or unix:/tmp/torchbeast).",
)


def main(argv):
    if FLAGS.role == "actor":
        actor.main(argv)
    elif FLAGS.role == "learner":
        learner.main(argv)
    else:
        raise ValueError("Unknown role %s" % FLAGS.role)


if __name__ == "__main__":
    app.run(main)
