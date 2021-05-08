# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    # Return a tensor of shape (T, N) with the log-likelihood of each action
    # found in the `actions` tensor (also of shape (T, N)).
    return -F.nll_loss(
        F.log_softmax(torch.flatten(policy_logits, 0, 1), dim=-1),
        torch.flatten(actions, 0, 1),
        reduction="none",
    ).view_as(actions)


def from_logits(
    behavior_policy_logits,  # (T, N, num_actions)
    target_policy_logits,  # (T, N, num_actions)
    actions,  # (T, N)
    discounts,  # (T, N)
    rewards,  # (T, N)
    values,  # (T, N)
    bootstrap_value,  # (N)
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    end_of_episode_bootstrap=False,
    done=None,  # (T, N)
):
    """V-trace for softmax policies."""
    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs  # (T, N)
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        end_of_episode_bootstrap=end_of_episode_bootstrap,
        done=done,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
    log_rhos,  # (T, N)
    discounts,  # (T, N)
    rewards,  # (T, N)
    values,  # (T, N)
    bootstrap_value,  # (N)
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    end_of_episode_bootstrap=False,
    done=None,  # (T, N)
):
    """V-trace from log importance weights."""
    if end_of_episode_bootstrap:
        assert done is not None  # we need the end of episode markers to bootstrap
    with torch.no_grad():  # TODO could be removed since we use `@torch.no_grad()`
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )  # (T, N)
        deltas = clipped_rhos * (
            rewards + discounts * values_t_plus_1 - values
        )  # (T, N)

        if end_of_episode_bootstrap:
            # Set the TD delta to 0 for last step of episode. This is so that
            # when computing `vs` below, its value at the previous step k will be
            # equal to rho_k * (r_k + gamma V_k+1 - V_k), which achieves the desired
            # bootstrapping effect.
            deltas[done] = 0.0

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):  # T-1, T-2, ..., 0
            # This is eq. 1 in IMPALA's paper https://arxiv.org/abs/1802.01561
            # but without the V(x_s) (which will be added back below). It is
            # computed recursively as in "Remark 1" in the paper.
            # Note that a different `n` (in the "n-steps V-trace target") is
            # used for each entry in the rollout: the last entry uses n=1, while
            # the first one uses n=T.
            acc = deltas[t] + discounts[t] * cs[t] * acc  # (N)
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)  # (T, N)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)  # (T, N)

        # V values at the next timestep (using boostrap value for end of rollout).
        # Note that if `end_of_episode_bootstrap` is True, then `vs` at last step of
        # an episode is equal to V, which is the correct value to use as `vs_t_plus_1`
        # for the step before.
        vs_t_plus_1 = torch.cat(
            [vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )  # (T, N)

        # Advantage for policy gradient.
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        # In IMPALA's paper this is the term multiplying the gradient of log pi
        # (see end of Section 4).
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        if end_of_episode_bootstrap:
            # We cannot learn anything from the action taken at the last step of the
            # episode, because we do not have access to the next state (which is required
            # for boostrapping). Thus we set the advantage to zero so as to kill gradient.
            pg_advantages = pg_advantages * (~done).float()

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
