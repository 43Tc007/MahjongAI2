#!/usr/bin/env python3
"""
Standalone test script for Mahjong policy against random agents.
Usage: python test_agent.py --model_path policy_net_first.pth --episodes 100
"""

import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
import argparse
from tqdm.auto import tqdm

# TorchRL imports
from torchrl.envs.utils import step_mdp
from torchrl.modules import ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictModule as TDM  # avoid name clash

# Your environment
from env_simplified import make_env

# ---------------------------------------------------------------------
# 1. Model architecture (copied from your notebook)
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
    def __init__(self, channels, dilation=1, *, norm_builder=nn.Identity,
                 actv_builder=nn.ReLU, pre_actv=False):
        super().__init__()
        self.pre_actv = pre_actv
        pad = dilation
        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(), actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(), actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(), actv_builder(),
                nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, conv_channels, num_blocks, seq_len, *,
                 actv_builder=nn.Mish, pre_actv=True):
        super().__init__()
        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i
            blocks.append(ResBlock(
                conv_channels, dilation=dilation,
                norm_builder=norm_builder, actv_builder=actv_builder, # type: ignore
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
    def __init__(self, in_channels: int, action_space: int, seq_len: int,
                 conv_channels: int = 256, num_blocks: int = 4):
        super().__init__()
        actv_builder = partial(nn.Mish, inplace=True)
        self.encoder = ResNet(
            in_channels=in_channels,
            conv_channels=conv_channels,
            num_blocks=num_blocks,
            seq_len=seq_len,
            actv_builder=actv_builder,# type: ignore
            pre_actv=True,
        )
        self.actv = actv_builder()
        self.fc_q = nn.Linear(1024, action_space)

    def forward(self, obs: Tensor) -> Tensor:
        phi = self.encoder(obs)
        phi = self.actv(phi)
        return self.fc_q(phi)


class MultiAgentAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
            batch, agents, C, L = x.shape
            x_flat = x.view(batch * agents, C, L)
            out_flat = self.model(x_flat)
            out = out_flat.view(batch, agents, -1)
            return out.squeeze(0)
        elif x.dim() == 4:
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


# ---------------------------------------------------------------------
# 2. Helper: random action from mask
# ---------------------------------------------------------------------

def random_action_from_mask(action_mask):
    """
    action_mask: 1D torch.Tensor of shape (n_actions,), 1=valid, 0=invalid.
    Returns: int action index.
    """
    valid = torch.nonzero(action_mask, as_tuple=True)[0]
    if len(valid) == 0:
        return 0  # fallback (should not happen)
    return valid[torch.randint(0, len(valid), (1,))].item()


# ---------------------------------------------------------------------
# 3. Test function
# ---------------------------------------------------------------------

def test_policy_against_random(env, policy, num_episodes=100, device='cpu'):
    """
    Runs episodes with agent 0 using `policy`, others random.
    Returns dict of metrics.
    """
    policy.eval()
    policy = policy.to(device)

    wins = losses = draws = 0
    total_returns = []
    terminal_rewards = []

    for _ in tqdm(range(num_episodes), desc="Testing", leave=False):
        td = env.reset()
        done = False
        episode_return = 0.0
        terminal_reward = 0.0

        while not done:
            # 1. Policy forward
            td = policy(td)

            # 2. Override actions for agents 1,2,3 with random
            actions = td[('agents', 'action')]
            if actions.dim() == 2:
                # shape: (batch, agents) – we assume batch=1 during testing
                for agent_id in [1, 2, 3]:
                    mask = td[('agents', 'action_mask')][0, agent_id]
                    actions[0, agent_id] = random_action_from_mask(mask)
            else:
                # shape: (agents,)
                for agent_id in [1, 2, 3]:
                    mask = td[('agents', 'action_mask')][agent_id]
                    actions[agent_id] = random_action_from_mask(mask)

            # 3. Environment step
            td = env.step(td)
            done = td[('next', 'done')].any().item()
            if done:
                terminal_reward = (td[('next', 'agents', 'reward')][0, 0].item())

            # 5. Advance to next timestep
            td = step_mdp(td)

        terminal_rewards.append(terminal_reward)

        if terminal_reward > 0:
            wins += 1
        elif terminal_reward < 0:
            losses += 1
        else:
            draws += 1

    n = num_episodes
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "avg_terminal_reward": sum(terminal_rewards) / n,
    }


# ---------------------------------------------------------------------
# 4. Main: load model and run test
# ---------------------------------------------------------------------

def build_policy(device='cpu'):
    """
    Builds the policy network with the same architecture as used in training.
    """
    # Match the parameters from your notebook:
    in_channels = 34      # observation channels for each agent
    action_space = 75     # number of possible actions
    seq_len = 29          # sequence length (input shape)

    base_net = UnifiedQNetwork(
        in_channels=in_channels,
        action_space=action_space,
        seq_len=seq_len,
        conv_channels=256,
        num_blocks=4,
    )
    # Adapt for multi-agent
    policy_net = nn.Sequential(
        CastToFloat(),
        MultiAgentAdapter(base_net)
    )
    # TensorDictModule
    policy_module = TDM(
        policy_net,
        in_keys=[("agents", "observation", "observation")],
        out_keys=[("agents", "logits")],
    )
    # ProbabilisticActor (needs action spec and mask keys)
    # We need an environment to get the action_spec_unbatched – we'll build a dummy env.
    # But we can also hardcode the spec. Simpler: create env, get spec.
    env = make_env()
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys={'logits': ('agents', 'logits'), 'mask': ('agents', 'action_mask')},  # type: ignore
        out_keys=[env.action_key],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    return policy, env


def main():
    parser = argparse.ArgumentParser(description="Test Mahjong policy against random agents.")
    parser.add_argument('--model_path', type=str, default='policy_net_first.pth',
                        help='Path to the saved policy network state_dict (or full model).')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of test episodes.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on (cpu or cuda).')
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build policy and environment
    policy, env = build_policy(device=device) 
    # The saved file may be a full model or state_dict.
    # From your notebook you saved the full model object, but then loaded state_dict.
    # We'll try both: if it's a state_dict, load it directly; if it's a full model, extract state_dict.
    try:
        saved = torch.load(args.model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if hasattr(saved, 'state_dict'):
        # It's a full model object
        state_dict = saved.state_dict()
    else:
        # It's a state_dict (dict)
        state_dict = saved

    # Load into the policy's underlying network.
    # The policy is a ProbabilisticActor; its module is policy_module, which contains the nn.Sequential.
    # The state_dict keys correspond to the nested modules.
    policy.module.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    # Run test
    metrics = test_policy_against_random(env, policy, num_episodes=args.episodes, device=device)

    print("\n===== Test Results =====")
    print(f"Episodes: {args.episodes}")
    print(f"Win rate:  {metrics['win_rate']:.3f}")
    print(f"Loss rate: {metrics['loss_rate']:.3f}")
    print(f"Draw rate: {metrics['draw_rate']:.3f}")
    print(f"Avg terminal reward: {metrics['avg_terminal_reward']:.3f}")


if __name__ == "__main__":
    main()