import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from typing import List, Tuple

from model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-3  # learning rate of the critic 
WEIGHT_DECAY = 0  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size: int, action_size: int, random_seed: int):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = BATCH_SIZE

        # Construct Actor networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Construct Critic networks 
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise processing
        self.noise = OUNoise(action_size, random_seed)

    def step(self):
        if len(SHARED_BUFFER) > BATCH_SIZE:
            experiences = SHARED_BUFFER.sample(device)
            self.learn(experiences, GAMMA)

    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences: Tuple[List[torch.Tensor], ...], gamma: float):
        """Update policy and value parameters using given batch of experience tuples.

        Args:
            experiences (Tuple[List[torch.Tensor], ...]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_list, actions_list, rewards, next_states_list, dones = experiences

        next_states_tensor = torch.cat(next_states_list, dim=1).to(device)
        states_tensor = torch.cat(states_list, dim=1).to(device)
        actions_tensor = torch.cat(actions_list, dim=1).to(device)

        # ---------------------------- Update Critic ---------------------------- #
        next_actions = [self.actor_target(next_states) for next_states in next_states_list]
        next_actions_tensor = torch.cat(next_actions, dim=1).to(device)
        Q_targets_next = self.critic_target(next_states_tensor, next_actions_tensor)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Ensure Q_targets có kích thước [BATCH_SIZE, 1]
        # Sử dụng phép trung bình để giảm kích thước từ [256, 2] xuống [256, 1]
        Q_targets = Q_targets.mean(dim=1, keepdim=True)

        # Compute Q expected
        Q_expected = self.critic_local(states_tensor, actions_tensor)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        actions_pred = [self.actor_local(states) for states in states_list]
        actions_pred_tensor = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states_tensor, actions_pred_tensor).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- Update Target Networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)





    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module, tau: float):
        """Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params:
            local_model (torch.nn.Module): model (weights will be copied from)
            target_model (torch.nn.Module): model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class MADDPG:
    """Multi-Agent DDPG"""

    def __init__(self, state_size: int, action_size: int, random_seed: int, num_agents: int = 2):
        self.num_agents = num_agents
        self.action_size = action_size
        self.agents = [DDPGAgent(state_size, action_size, random_seed) for _ in range(self.num_agents)]

    def step(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
             dones: np.ndarray):
        SHARED_BUFFER.add(states, actions, rewards, next_states, dones)
        for agent in self.agents:
            agent.step()

    def act(self, states: np.ndarray, add_noise: bool = True) -> np.ndarray:
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions

    def save_weights(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), f'agent{index + 1}_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'agent{index + 1}_checkpoint_critic.pth')

    def reset(self):
        for agent in self.agents:
            agent.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size: int, seed: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])

    def add(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_states: np.ndarray,
            dones: np.ndarray):
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)

    def sample(self, device: torch.device, num_agents: int = 2) -> Tuple[List[torch.Tensor], ...]:
        experiences = random.sample(self.memory, k=self.batch_size)

        states_list = [
            torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for
            index in range(num_agents)]
        actions_list = [
            torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for
            index in range(num_agents)]
        next_states_list = [
            torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device)
            for index in range(num_agents)]
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states_list, actions_list, rewards, next_states_list, dones

    def __len__(self) -> int:
        return len(self.memory)


# Replay buffer that will be shared by the two agents
SHARED_BUFFER = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
