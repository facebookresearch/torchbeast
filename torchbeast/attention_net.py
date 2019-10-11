from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

"""
AttentionNet is a pytorch reimpl. of the following paper for the NeurIPS reproducibility challenge.

@article{mott2019towards,
  title={Towards Interpretable Reinforcement Learning Using Attention Augmented Agents},
  author={Mott, Alex and Zoran, Daniel and Chrzanowski, Mike and Wierstra, Daan and Rezende, Danilo J},
  journal={arXiv preprint arXiv:1906.02500},
  year={2019}
}
"""

class AttentionNet(nn.Module):
    def __init__(
        self,
        __obseravation_num_channels: int,
        num_actions: int,
        __use_lstm: bool,
        hidden_size: int = 256,
        c_v: int = 120,
        c_k: int = 8,
        c_s: int = 64,
        num_queries: int = 4,
    ):
        """AttentionNet implementing the attention agent.
        """
        super(AttentionNet, self).__init__()

        self.num_queries = num_queries
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.c_v, self.c_k, self.c_s, self.num_queries = c_v, c_k, c_s, num_queries

        self.vision = VisionNetwork()
        self.query = QueryNetwork()
        self.spatial = SpatialBasis()

        self.answer_processor = nn.Sequential(
            # 1043 x 512
            nn.Linear(
                (c_v + c_s) * num_queries
                + (c_k + c_s) * num_queries
                + 1
                + self.num_actions,
                512,
            ),
            nn.ReLU(),
            nn.Linear(512, hidden_size),
        )

        # self.policy_core = nn.LSTMCell(hidden_size, hidden_size)
        self.policy_core = nn.LSTMCell(hidden_size, hidden_size)
        self.policy_head = nn.Sequential(nn.Linear(hidden_size, num_actions))
        self.values_head = nn.Sequential(nn.Linear(hidden_size, 1))

    def initial_state(self, batch_size):
        # NOTE: Hard-coded values.
        core_zeros = torch.zeros(batch_size, self.hidden_size).float()
        height, width = 17, 17
        conv_zeros = torch.zeros(
            batch_size, self.hidden_size // 2, height, width
        ).float()
        return (
            core_zeros.clone(),  # hidden
            core_zeros.clone(),  # cell
            conv_zeros.clone(),  # hidden
            conv_zeros.clone(),  # cell
        )

    def forward(self, inputs, prev_state):

        # 1 (a). Vision.
        # --------------

        # [T, B, C, H, W].
        X = inputs["frame"]
        T, B, *_ = X.size()

        vision_state = splice_vision_state(prev_state)
        # -> [N, c_k+c_v, h, w] s.t. N = T * B.
        O, next_vision_state = self.vision(X, inputs, vision_state)
        # -> [N, h, w, c_k+c_v]
        O = O.transpose(1, 3)

        # -> [N, h, w, c_k], [N, h, w, c_v]
        K, V = O.split([self.c_k, self.c_v], dim=3)
        # -> [N, h, w, c_k + c_s], [N, h, w, c_v + c_s]
        K, V = self.spatial(K), self.spatial(V)

        # 2. Prepare all inputs to operate over the T steps.
        # --------------------------------------------------
        core_output_list = []
        core_state = splice_core_state(prev_state)
        prev_output = core_state[0]
        _, h, w, _ = O.size()

        # [N, h, w, num_keys = c_k + c_s] -> [T, B, h, w, num_keys = c_k + c_s]
        K = K.view(T, B, h, w, -1)
        # [N, h, w, num_values = c_v + c_s] -> [T, B, h, w, num_values = c_v + c_s]
        V = V.view(T, B, h, w, -1)
        # -> [T, B, 1]
        notdone = (1 - inputs["done"].float()).view(T, B, 1)
        # -> [T, B, 1, num_actions]
        prev_action = (
            F.one_hot(inputs["last_action"].view(T * B), self.num_actions)
            .view(T, B, 1, self.num_actions)
            .float()
        )
        # -> [T, B, 1, 1]
        prev_reward = torch.clamp(inputs["reward"], -1, 1).view(T, B, 1, 1)

        # 3. Operate over the T time steps.
        # ---------------------------------
        # NOTE: T = 1 when 'act'ing and T > 1 when 'learn'ing.

        for K_t, V_t, prev_reward_t, prev_action_t, nd_t in zip(
            K.unbind(),
            V.unbind(),
            prev_reward.unbind(),
            prev_action.unbind(),
            notdone.unbind(),
        ):

            # A. Queries.
            # --------------
            # [B, hidden_size] -> [B, num_queries, num_keys]
            Q_t = self.query(prev_output)
            # NOTE: The queries and keys don't have intrinsic meaning -- they're trained
            # to be relevant. Furthermore, the sizing relies on selecting `smart` hidden
            # and output sizes in the QueryNetwork.

            # B. Answer.
            # ----------
            # [B, h, w, num_keys] x [B, 1, num_keys, num_queries] -> [B, h, w, num_queries]
            A = torch.matmul(K_t, Q_t.transpose(2, 1).unsqueeze(1))
            # -> [B, h, w, num_queries]
            A = spatial_softmax(A)
            # [B, h, w, num_queries] x [B, h, w, num_values] -> [B, num_queries, num_values]
            answers = apply_alpha(A, V_t)

            # -> [B, Z = {num_values * num_queries + num_keys * num_queries + 1 + num_actions}]
            chunks = list(
                torch.chunk(answers, self.num_queries, dim=1)
                + torch.chunk(Q_t, self.num_queries, dim=1)
                + (prev_reward_t.float(), prev_action_t.float())
            )
            answer = torch.cat(chunks, dim=2).squeeze(1)
            # [B, Z] -> [B, hidden_size]
            core_input = self.answer_processor(answer)

            # C. Policy.
            # ----------

            # NOTE: nd_t is 0 when episode is done, 1 otherwise. Thus, this resets the state
            # as episodes finish.
            core_state = tuple((nd_t * s) for s in core_state)

            # -> [B, hidden_size]
            prev_output, _ = core_state = self.policy_core(core_input, core_state)
            core_output_list.append(prev_output)
        next_core_state = core_state
        # -> [T * B, hidden_size]
        output = torch.cat(core_output_list, dim=0)

        # 4. Outputs.
        # --------------
        # [T * B, hidden_size] -> [T * B, num_actions]
        logits = self.policy_head(output)
        # [T * B, hidden_size] -> [T * B, 1]
        values = self.values_head(output)

        if self.training:
            # [T * B, num_actions] -> [T * B, 1]
            action = torch.multinomial(F.softmax(logits, dim=1), num_samples=1)
        else:
            # [T * B, num_actions] -> [T * B, 1]
            # Don't sample when testing.
            action = torch.argmax(logits, dim=1)

        # Format for torchbeast.
        # [T * B, num_actions] -> [T * B, num_actions]
        policy_logits = logits.view(T, B, self.num_actions)
        # [T * B, 1] -> [T, B]
        baseline = values.view(T, B)
        # [T * B, 1] -> [T, B]
        action = action.view(T, B)

        # Create tuple of next states.
        next_state = next_core_state + next_vision_state
        return (
            dict(policy_logits=policy_logits, baseline=baseline, action=action),
            next_state,
        )

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        """Initialize stateful ConvLSTM cell.
        
        Parameters
        ----------
        input_channels : ``int``
            Number of channels of input tensor.
        hidden_channels : ``int``
            Number of channels of hidden state.
        kernel_size : ``int``
            Size of the convolutional kernel.
            
        Paper
        -----
        https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf
        
        Referenced code
        ---------------
        https://github.com/automan000/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py        
        """
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whi = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxf = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whf = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxc = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Whc = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )
        self.Wxo = nn.Conv2d(
            self.input_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=True,
        )
        self.Who = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            1,
            self.padding,
            bias=False,
        )

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, state):
        if self.Wci is None:
            batch_size, _, height, width = x.size()
            self.initial_state(
                batch_size, self.hidden_channels, height, width, x.device
            )
        h, c = state
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)

        state = ch, cc
        return ch, state

    def initial_state(self, batch_size, hidden, height, width, device):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wcf = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )
            self.Wco = torch.zeros(1, hidden, height, width, requires_grad=True).to(
                device
            )


class VisionNetwork(nn.Module):
    def __init__(self):
        super(VisionNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=1),
        )
        self.lstm = ConvLSTMCell(input_channels=64, hidden_channels=128, kernel_size=3)

    def forward(self, X, inputs, vision_lstm_state):
        T, B, *_ = X.size()
        X = torch.flatten(X, 0, 1).float() / 255.0
        X = self.cnn(X)
        _, C, H, W = X.size()
        X = X.view(T, B, C, H, W)
        output_list = []
        notdone = 1 - inputs["done"].float()
        # Operating over T:
        for X_t, nd in zip(X.unbind(), notdone.unbind()):
            nd = nd.view(B, 1, 1, 1)
            vision_lstm_state = tuple((nd * s) for s in vision_lstm_state)
            O_t, vision_state = self.lstm(X_t, vision_lstm_state)
            output_list.append(O_t)
        next_vision_state = vision_state
        # (T * B, hidden_size)
        O = torch.cat(output_list, dim=0)
        return O, next_vision_state


class QueryNetwork(nn.Module):
    def __init__(self):
        super(QueryNetwork, self).__init__()
        # TODO: Add proper non-linearity.
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 288),
            nn.ReLU(),
            nn.Linear(288, 288),
        )

    def forward(self, query):
        out = self.model(query)
        return out.reshape(-1, 4, 72)


class SpatialBasis:
    # TODO: Implement Spatial.
    """
    NOTE: The `height` and `weight` depend on the inputs' size and its resulting size
    after being processed by the vision network.
    """

    def __init__(self, height=17, width=17, channels=64):
        h, w, d = height, width, channels

        p_h = torch.mul(
            torch.arange(1, h + 1).unsqueeze(1).float(), torch.ones(1, w).float()
        ) * (np.pi / h)
        p_w = torch.mul(
            torch.ones(h, 1).float(), torch.arange(1, w + 1).unsqueeze(0).float()
        ) * (np.pi / w)

        # TODO: I didn't quite see how U,V = 4 made sense given that the authors form the spatial
        # basis by taking the outer product of the values.
        # I am not confident in this step.
        U = V = 8  # size of U, V.
        u_basis = v_basis = torch.arange(1, U + 1).unsqueeze(0).float()
        a = torch.mul(p_h.unsqueeze(2), u_basis)
        b = torch.mul(p_w.unsqueeze(2), v_basis)
        out = torch.einsum("hwu,hwv->hwuv", torch.cos(a), torch.cos(b)).reshape(h, w, d)
        self.S = out

    def __call__(self, X):
        # Stack the spatial bias (for each batch) and concat to the input.
        batch_size = X.size()[0]
        S = torch.stack([self.S] * batch_size).to(X.device)
        return torch.cat([X, S], dim=3)


def spatial_softmax(A):
    """Softmax over the attention map.
    """
    b, h, w, d = A.size()
    # Flatten A s.t. softmax is applied to each grid (height by width) (not over queries or channels).
    A = A.reshape(b, h * w, d)
    A = F.softmax(A, dim=1)
    # Reshape A back to original shape.
    A = A.reshape(b, h, w, d)
    return A


def apply_alpha(A, V):
    # TODO: Check this function again.
    b, h, w, c = A.size()
    
    # [B, h, w, c] -> [B, h * w, c]
    A = A.reshape(b, h * w, c)
    # [B, h * w, c] -> [B, c, h * w] 
    A = A.transpose(1, 2)

    d = V.size(3)
    # [B, h, w, d] -> [B, h * w, d] 
    V = V.reshape(b, h * w, d)

    # [B, h * w, d] x [B, c, h * w] -> [B, 1, c] 
    return torch.matmul(A, V)


def splice_core_state(state: Tuple):
    return state[0], state[1]


def splice_vision_state(state: Tuple):
    return state[2], state[3]

