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

from absl import flags

from atari import atari_wrappers

FLAGS = flags.FLAGS

# Environment settings.
flags.DEFINE_string("env", "Pong", "Gym environment.")


def create_env():

    env_version = "v4"  # "v0" for "sticky actions".
    full_env_name = f"{FLAGS.env}NoFrameskip-{env_version}"
    # env = gym.make(full_env_name, full_action_space=True)

    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(full_env_name),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )
