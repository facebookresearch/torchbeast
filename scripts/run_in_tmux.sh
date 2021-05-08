#!/bin/bash
#
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
# Run via ./scripts/run_in_tmux.sh
#

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/..

WORKERS=4

CONDA_ENV=${CONDA_DEFAULT_ENV:-torchbeast}
CONDA_CMD="conda activate ${CONDA_ENV}"

WINDOW_NAME=torchbeast-run

tmux new-window -d -n "${WINDOW_NAME}"

TORCHBEAST_BINARY="python -m atari.main"

COMMAND='rm -rf /tmp/torchbeast && mkdir /tmp/torchbeast && '"${TORCHBEAST_BINARY}"' --role learner --num_actors='"${WORKERS}"' --batch_size 8 --total_steps 100000 --learning_rate 0.001 --log_dir /tmp/torchbeast --alsologtostderr '"$@"' '

tmux send-keys -t "${WINDOW_NAME}" "${CONDA_CMD}" KPEnter
tmux send-keys -t "${WINDOW_NAME}" "${COMMAND}" KPEnter

tmux split-window -t "${WINDOW_NAME}"

tmux send-keys -t "${WINDOW_NAME}" "${CONDA_CMD}" KPEnter

for ((id=0; id<$WORKERS; id++)); do
    COMMAND=''"${TORCHBEAST_BINARY}"' --role actor --actor_id '"${id}"' --log_dir /tmp/torchbeast --alsologtostderr '"$@"' &'
    tmux send-keys -t "${WINDOW_NAME}" "$COMMAND" ENTER
done
