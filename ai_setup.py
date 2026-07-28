import os
import torch
from torch import nn, Tensor
from functools import partial
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, MaskedCategorical

# ---------------------------------------------------------------------
# 1. Core building blocks (unchanged)
# ---------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        x = weight.unsqueeze(-1) * x
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dilation=1,  # <-- NEW: dilation factor
        *,
        norm_builder=nn.Identity,
        actv_builder=nn.ReLU,
        pre_actv=False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        # Padding must equal dilation to keep the sequence length unchanged
        pad = dilation 

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(),
                actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    # forward() remains identical
    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        seq_len,
        *,
        actv_builder=nn.Mish,
        pre_actv=True,
    ):
        super().__init__()

        # FIX: explicitly use conv_channels here
        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)

        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(ResBlock(
                conv_channels,
                dilation=dilation,
                norm_builder=norm_builder, # type: ignore
                actv_builder=actv_builder, # type: ignore
                pre_actv=pre_actv,
            ))

        layers = [nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        layers += [
            nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(32 * seq_len, 1024),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnifiedQNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_space: int,
        seq_len: int,
        conv_channels: int = 256,
        num_blocks: int = 4,
    ):
        super().__init__()
        actv_builder = partial(nn.Mish, inplace=True)
        self.encoder = ResNet(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            seq_len=seq_len,
            actv_builder=actv_builder, # type: ignore
            pre_actv=True,
        )
        self.actv = actv_builder()
        self.fc_q = nn.Linear(1024, action_space)
        self._freeze_bn = False

    def forward(self, obs: Tensor) -> Tensor:
        phi = self.encoder(obs)
        phi = self.actv(phi)
        return self.fc_q(phi)

# ---------------------------------------------------------------------

class MultiAgentAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.dim() == 3:
            # Input: (agents, C, L) – no batch dimension
            x = x.unsqueeze(0)                    # (1, agents, C, L)
            batch, agents, C, L = x.shape
            x_flat = x.view(batch * agents, C, L)
            out_flat = self.model(x_flat)        # (batch*agents, action_space)
            out = out_flat.view(batch, agents, -1)
            return out.squeeze(0)                # (agents, action_space)
        elif x.dim() == 4:
            # Input: (batch, agents, C, L)
            batch, agents, C, L = x.shape
            x_flat = x.view(batch * agents, C, L)
            out_flat = self.model(x_flat)
            out = out_flat.view(batch, agents, -1)
            return out
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")

class CastToFloat(nn.Module):
    def forward(self, x):
        return x.float()
    
class StateAdapter(nn.Module):
        def __init__(self, base_net, in_channels, agents=4):
            super().__init__()
            self.base_net = base_net
            self.in_channels = in_channels
            self.agents = agents

        def forward(self, x):
            added_batch = False
            if x.dim() == 2:                     # (C, L) -> add batch
                x = x.unsqueeze(0)               # (1, C, L)
                added_batch = True

            # Ensure correct channel order: (batch, channels, length)
            if x.shape[1] != self.in_channels:
                x = x.transpose(1, 2)            # (batch, L, C) -> (batch, C, L)

            out = self.base_net(x)               # (batch, agents)

            if added_batch:
                out = out.squeeze(0)             # (agents,)

            # Ensure last dimension is 1 (for state_value)
            if out.dim() == 1:
                out = out.unsqueeze(-1)          # (agents, 1)
            elif out.dim() == 2:
                out = out.unsqueeze(-1)          # (batch, agents, 1)
            return out
# =====================================================================
# 3. Instantiate policy and critic with correct shapes
# =====================================================================

def make_policy_critic(env, policy_model_path=None, critic_model_path=None):
    # ------------------- POLICY -------------------
    base_net = UnifiedQNetwork(
        in_channels=34,
        action_space=75,
        seq_len=29,
        conv_channels=256,
        num_blocks=4,
    )
    policy_net = nn.Sequential(
        CastToFloat(),
        MultiAgentAdapter(base_net)
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation", "observation")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys={
            'logits': ('agents', 'logits'),
            'mask': ('agents', 'action_mask')
        }, # type: ignore
        out_keys=[env.action_key],
        distribution_class=MaskedCategorical,
        return_log_prob=True
    )

    
    # ------------------- CRITIC -------------------
    base_critic = UnifiedQNetwork(
        in_channels=42,
        action_space=4,
        seq_len=52,
        conv_channels=256,
        num_blocks=4,
    )

    critic_net = nn.Sequential(
        CastToFloat(),
        StateAdapter(base_critic, in_channels=42)
    )
    critic = TensorDictModule(
        module=critic_net,
        in_keys=["state"],
        out_keys=[("agents", "state_value")],
    )
    # Load the full model object if valid paths were provided.
    if policy_model_path and critic_model_path:
        if os.path.exists(policy_model_path) and os.path.exists(critic_model_path):
            loaded_policy = torch.load(policy_model_path)
            policy.load_state_dict(loaded_policy)

            loaded_critic = torch.load(critic_model_path)
            critic.load_state_dict(loaded_critic)
            print('Model loaded successfully')
        else:
            print(f"Skipping checkpoint load because one or both files were not found: {policy_model_path}, {critic_model_path}")
    else:
        print('Using random models')
    return policy, critic
