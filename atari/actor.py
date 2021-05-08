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
from absl import logging

import numpy as np

from atari import environment
import torchbeast


FLAGS = flags.FLAGS

flags.DEFINE_integer("actor_id", 0, "Actor id.")
flags.DEFINE_bool("render_actor", False, "If actor should be rendered.")


def actor_loop():
    actor_step = 0
    episodes = 0

    while True:
        try:
            env = environment.create_env()

            client = torchbeast.Client(FLAGS.server_address)
            client.connect(10)

            run_id = np.random.randint(np.iinfo(np.int64).max)
            logging.info("Actor %i starting run %i", FLAGS.actor_id, run_id)

            observation = env.reset()
            reward = 0.0
            done = False

            episode_return = 0.0

            while True:
                action = client.inference(
                    FLAGS.actor_id, run_id, (observation, reward, done)
                )
                if FLAGS.render_actor:
                    env.render()
                observation, reward, done, info = env.step(action)

                actor_step += 1
                episode_return += reward

                if done:
                    observation = env.reset()
                    episodes += 1
                    if episodes % 100 == 0:
                        logging.info(
                            "After %i episodes (%i steps) return %f",
                            episodes,
                            actor_step,
                            episode_return,
                        )
                    episode_return = 0.0

        except (ConnectionError, TimeoutError) as e:
            if FLAGS.actor_id == 0:  # Let most actors fail silently.
                logging.exception(e)
            env.close()
            break
        except KeyboardInterrupt:
            logging.info("Stopping.")
            break


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    if FLAGS.log_dir:
        logging.get_absl_handler().use_absl_log_file(
            program_name=f"{FLAGS.role}-{FLAGS.actor_id}"
        )
    actor_loop()


if __name__ == "__main__":
    app.run(main)
