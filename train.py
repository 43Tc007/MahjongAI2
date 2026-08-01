import torch
from torch import nn
from torchrl.collectors import MultiSyncCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from env_simplified import make_env
from ai_setup import make_policy_critic
from torchrl.envs.utils import ExplorationType
import os
import argparse

def train_PPO(policy_path, critic_path, n_iters=100, auto_shutdown=False, save_checkpoint=True, save_final=True, num_workers=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    env = make_env()
    policy, critic = make_policy_critic(env, policy_path, critic_path)

    policy = policy.to(device)
    loss_module = ClipPPOLoss(
        actor_network=policy, # type: ignore
        critic_network=critic,
        entropy_coeff=0.01
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
    lr = 1e-4
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )  
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    loss_module = loss_module.to(device)

    num_epochs = 5
    max_grad_norm = 0.15
    frames_per_batch = 2048  # Number of team frames collected per training iteration
    total_frames = frames_per_batch * n_iters
    minibatch_size = 256

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this siz
    )
    policy=policy.to(device)
    collector = MultiSyncCollector(
        [make_env] * num_workers,
        policy=policy,
        device='cpu',
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        cat_results=0,
        exploration_type=ExplorationType.RANDOM # type: ignore
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
        if it % 60 == 0 and save_checkpoint:
            torch.save(policy.state_dict(), 'policy_checkpoint.pth')
            torch.save(critic.state_dict(), 'critic_checkpoint.pth')

    if save_final:
        torch.save(policy.state_dict(), 'policy_fin.pth')
        torch.save(critic.state_dict(), 'critic_fin.pth')
    if auto_shutdown:
        # shutil.move('/home/user/Documents/report.txt', '/home/user/Backup/')
        # os.system("/usr/bin/shutdown")
        os.system("shutdown /s /t 0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, default='', help="Number of training iterations")
    parser.add_argument("--critic_path", type=str, default='', help="Policy path")
    parser.add_argument("--n_iters", type=int, default=400, help="Critic path")
    parser.add_argument("--auto_shutdown", type=int, default=0, help="Shutdown after training")
    parser.add_argument("--save_checkpoint", type=int, default=1, help="save_checkpoint")
    parser.add_argument("--save_final", type=int, default=1, help="save_final")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    args = parser.parse_args()
    train_PPO(
        policy_path=args.policy_path, 
        critic_path=args.critic_path, 
        n_iters=args.n_iters, 
        auto_shutdown=bool(args.auto_shutdown), 
        save_checkpoint=bool(args.save_checkpoint), 
        save_final=bool(args.save_final)
    )