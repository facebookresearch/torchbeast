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

import torch
from torch import nn
from torch.nn import functional as F

import nest

FLAGS = flags.FLAGS

flags.DEFINE_bool("use_lstm", False, "Use LSTM in model.")
flags.DEFINE_enum("model", "resnet", ["test", "shallow", "resnet"], "The model to run.")


class TestModel(nn.Module):
    def __init__(self, num_actions, core_output_size=256):
        super(TestModel, self).__init__()
        if FLAGS.use_lstm:
            raise ValueError("Test model cannot use LSTM.")
        self.num_actions = num_actions
        self.linear = nn.Linear(4 * 84 * 84, core_output_size)
        self.policy = nn.Linear(core_output_size, num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, last_actions, env_outputs, core_state, unroll=False):
        if not unroll:
            # [T=1, B, ...].
            env_outputs = nest.map(lambda t: t.unsqueeze(0), env_outputs)

        observation, reward, done = env_outputs
        T, B, *_ = observation.shape

        x = observation.reshape(T * B, -1)
        x = x.float() / 255.0

        core_output = self.linear(x)

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        outputs = dict(action=action, policy_logits=policy_logits, baseline=baseline)
        if not unroll:
            for t in outputs.values():
                t.squeeze_(0)
        return outputs, core_state


class ShallowModel(nn.Module):
    def __init__(self, num_actions, core_output_size=256):
        super(ShallowModel, self).__init__()

        self.num_actions = num_actions
        self.use_lstm = FLAGS.use_lstm

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layer.
        self.fc = nn.Linear(3136, 512)

        # FC output size + one-hot of last action + last reward.
        core_output_size = self.fc.out_features + num_actions + 1

        if self.use_lstm:
            self.core = nn.LSTMCell(core_output_size, 256)
            core_output_size = 256

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, last_actions, env_outputs, core_state, unroll=False):
        observation, reward, done = env_outputs

        if not unroll:
            # [T=1, B, ...].
            observation = observation.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        T = observation.shape[0]
        B = observation.shape[1]

        x = torch.flatten(observation, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            last_actions.view(T * B), self.num_actions
        ).float()
        reward = reward.view(T * B, 1).float()
        core_input = torch.cat([x, reward, one_hot_last_action], dim=1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            notdone.unsqueeze_(-1)  # [T, B, H=1] for broadcasting.

            for input_t, notdone_t in zip(core_input.unbind(), notdone.unbind()):
                core_state = nest.map(notdone_t.mul, core_state)
                output_t, core_state = self.core(input_t, core_state)
                core_state = (output_t, core_state)  # nn.LSTMCell is a bit weird.
                core_output_list.append(output_t)  # [[B, H], [B, H], ...].
            core_output = torch.cat(core_output_list)  # [T * B, H].
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        outputs = {
            "action": action,
            "policy_logits": policy_logits,
            "baseline": baseline,
        }

        if not unroll:
            for t in outputs.values():
                t.squeeze_(0)
        return outputs, core_state


class Model(nn.Module):
    def __init__(self, num_actions, use_lstm=None):
        super(Model, self).__init__()
        self.num_actions = num_actions
        self.use_lstm = use_lstm

        if use_lstm is None:
            self.use_lstm = FLAGS.use_lstm

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        self.convs = []

        input_channels = 4
        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        self.fc = nn.Linear(3872, 256)

        # FC output size + last reward + one-hot of last action.
        core_output_size = self.fc.out_features + 1 + num_actions

        if self.use_lstm:
            self.core = nn.LSTMCell(core_output_size, 256)
            core_output_size = 256

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, last_actions, env_outputs, core_state, unroll=False):
        if not unroll:
            # [T=1, B, ...].
            env_outputs = nest.map(lambda t: t.unsqueeze(0), env_outputs)

        observation, reward, done = env_outputs

        T, B, *_ = observation.shape
        x = torch.flatten(observation, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0

        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = x.view(T * B, -1)
        x = F.relu(self.fc(x))

        one_hot_last_action = F.one_hot(
            last_actions.view(T * B), self.num_actions
        ).float()
        reward = reward.view(T * B, 1).float()
        core_input = torch.cat([x, reward, one_hot_last_action], dim=1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~done).float()
            notdone.unsqueeze_(-1)  # [T, B, H=1] for broadcasting.

            for input_t, notdone_t in zip(core_input.unbind(), notdone.unbind()):
                core_state = nest.map(notdone_t.mul, core_state)
                output_t, core_state = self.core(input_t, core_state)
                core_state = (output_t, core_state)  # nn.LSTMCell is a bit weird.
                core_output_list.append(output_t)  # [[B, H], [B, H], ...].
            core_output = torch.cat(core_output_list)  # [T * B, H].
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            # Don't sample when testing.
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        outputs = dict(action=action, policy_logits=policy_logits, baseline=baseline)
        if not unroll:
            outputs = nest.map(lambda t: t.squeeze(0), outputs)
        return outputs, core_state


def get_model(num_actions):
    if FLAGS.model == "test":
        return TestModel(num_actions)
    if FLAGS.model == "shallow":
        return ShallowModel(num_actions)
    return Model(num_actions)
