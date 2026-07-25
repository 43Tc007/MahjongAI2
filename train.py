<<<<<<< HEAD
import torch
from torch import nn
from torchrl.envs import PettingZooWrapper
from env_simplified import MahjongGameEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import MarlGroupMapType
from torchrl.modules import MultiAgentConvNet, MultiAgentMLP, ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

env = PettingZooWrapper(
    env=MahjongGameEnv(),
    use_mask=True,
    return_state=True,
    categorical_actions=True,
    group_map=MarlGroupMapType.ALL_IN_ONE_GROUP
)
class CastToFloat(nn.Module):
    def forward(self, x):
        return x.float()   # or .to(torch.float32)

class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-3)

policy_net = nn.Sequential(
    CastToFloat(),
    Unsqueeze(),
    MultiAgentConvNet(
        n_agents=4,
        centralized=False,
        share_params=True,
        num_cells=[32, 32, 32],
        paddings=1,
        strides=1,
        kernel_sizes=3,
    ),
    MultiAgentMLP(
        n_agents=4,
        n_agent_inputs=None,
        n_agent_outputs=75,
        num_cells=256,
        centralized=False,
        share_params=True,
        depth=2,
    )
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
)  # we'll need the log-prob for the PPO loss

class UnsqueezeV2(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-3)
critic_net = nn.Sequential(
    CastToFloat(),
    UnsqueezeV2(),
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(32),
    nn.ReLU(),
    # nn.Dropout2d(0.5),  # Dropout2d is recommended for conv layers (channel-wise)
    
    # Block 2
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(64),
    nn.ReLU(),
    # nn.Dropout2d(0.5),
    
    # Block 3
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(128),
    nn.ReLU(),
    # nn.Dropout2d(0.5),
    
    # MLP Head (standard MultiAgentMLP style after flatten)
    nn.Flatten(-3),
    nn.Linear(69888, 512),  
    nn.ReLU(),
    nn.Linear(512, 4),
    nn.Unflatten(-1, (4, 1))
)
critic = TensorDictModule(
    module=critic_net,
    in_keys=["state"],               # global state
    out_keys=[("agents", "state_value")],  # shape [4] values under agents
)

policy = policy.to('cpu')
critic = critic.to('cpu')
print("Running policy:", policy(env.reset()))
print("Running value:", critic(env.reset()))

policy_net.load_state_dict(torch.load('actor_net_23072026_0.pth'))
critic.load_state_dict(torch.load("critic_net_23072026_0.pth"))
# policy_module= torch.compile(policy_module)
# critic = torch.compile(critic)

loss_module = ClipPPOLoss(
    actor_network=policy, # type: ignore
    critic_network=critic,
    entropy_coeff=0.02
)
loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agents", "done"),                # per-agent
    terminated=("agents", "terminated"),
)
gamma = 0.995  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
lr = 1e-5
loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

loss_module = loss_module.to(device)

num_epochs = 5
max_grad_norm = 0.1
frames_per_batch = 1000  # Number of team frames collected per training iteration
n_iters = 200  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters
minibatch_size = 1000

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this siz
)
policy=policy.to(device)
collector= Collector(
    env,
    policy,
    device='cpu',
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames
)

from tqdm.auto import tqdm
for it, tensordict_data in enumerate(tqdm(collector)):
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )  # Compute GAE and add it to the data

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            subdata = subdata.to(device)
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()


torch.save(policy_net.state_dict(), "actor_net_23072026_1.pth")
=======
import torch
from torch import nn
from torchrl.envs import PettingZooWrapper
from env_simplified import MahjongGameEnv
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import MarlGroupMapType
from torchrl.modules import MultiAgentConvNet, MultiAgentMLP, ProbabilisticActor, MaskedCategorical
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

env = PettingZooWrapper(
    env=MahjongGameEnv(),
    use_mask=True,
    return_state=True,
    categorical_actions=True,
    group_map=MarlGroupMapType.ALL_IN_ONE_GROUP
)
class CastToFloat(nn.Module):
    def forward(self, x):
        return x.float()   # or .to(torch.float32)

class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-3)

policy_net = nn.Sequential(
    CastToFloat(),
    Unsqueeze(),
    MultiAgentConvNet(
        n_agents=4,
        centralized=False,
        share_params=True,
        num_cells=[32, 32, 32],
        paddings=1,
        strides=1,
        kernel_sizes=3,
    ),
    MultiAgentMLP(
        n_agents=4,
        n_agent_inputs=None,
        n_agent_outputs=75,
        num_cells=256,
        centralized=False,
        share_params=True,
        depth=2,
    )
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
)  # we'll need the log-prob for the PPO loss

class UnsqueezeV2(nn.Module):
    def forward(self, x):
        return x.unsqueeze(-3)
critic_net = nn.Sequential(
    CastToFloat(),
    UnsqueezeV2(),
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(32),
    nn.ReLU(),
    # nn.Dropout2d(0.5),  # Dropout2d is recommended for conv layers (channel-wise)
    
    # Block 2
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(64),
    nn.ReLU(),
    # nn.Dropout2d(0.5),
    
    # Block 3
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
    # nn.BatchNorm2d(128),
    nn.ReLU(),
    # nn.Dropout2d(0.5),
    
    # MLP Head (standard MultiAgentMLP style after flatten)
    nn.Flatten(-3),
    nn.Linear(69888, 512),  
    nn.ReLU(),
    nn.Linear(512, 4),
    nn.Unflatten(-1, (4, 1))
)
critic = TensorDictModule(
    module=critic_net,
    in_keys=["state"],               # global state
    out_keys=[("agents", "state_value")],  # shape [4] values under agents
)

policy = policy.to('cpu')
critic = critic.to('cpu')
print("Running policy:", policy(env.reset()))
print("Running value:", critic(env.reset()))

policy_net.load_state_dict(torch.load('actor_net_23072026_0.pth'))
critic.load_state_dict(torch.load("critic_net_23072026_0.pth"))
# policy_module= torch.compile(policy_module)
# critic = torch.compile(critic)

loss_module = ClipPPOLoss(
    actor_network=policy, # type: ignore
    critic_network=critic,
    entropy_coeff=0.02
)
loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_key,
    action=env.action_key,
    value=("agents", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agents", "done"),                # per-agent
    terminated=("agents", "terminated"),
)
gamma = 0.995  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
lr = 1e-5
loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)  
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

loss_module = loss_module.to(device)

num_epochs = 5
max_grad_norm = 0.1
frames_per_batch = 1000  # Number of team frames collected per training iteration
n_iters = 200  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters
minibatch_size = 1000

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),
    batch_size=minibatch_size,  # We will sample minibatches of this siz
)
policy=policy.to(device)
collector= Collector(
    env,
    policy,
    device='cpu',
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames
)

from tqdm.auto import tqdm
for it, tensordict_data in enumerate(tqdm(collector)):
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )
    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

    with torch.no_grad():
        GAE(
            tensordict_data,
            params=loss_module.critic_network_params,
            target_params=loss_module.target_critic_network_params,
        )  # Compute GAE and add it to the data

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            subdata = subdata.to(device)
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()


torch.save(policy_net.state_dict(), "actor_net_23072026_1.pth")
>>>>>>> 3abff74c04fda3ad8932230acceef156f12c32ec
torch.save(critic.state_dict(), "critic_net_23072026_1.pth")