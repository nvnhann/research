import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorNetwork, CriticNetwork
import copy
from collections import namedtuple, deque

# Detect if CUDA is available and set device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Learning rate for the neural networks
LR = 3E-4

# Discount factor for future rewards
GAMMA = 0.99

# Batch size for the replay buffer
BATCH_SIZE = 64

# Soft update parameter
TAU = 1e-3

# Size of the replay buffer
BUFFER_SIZE = 1000000


class Agent:
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor networks (local and target)
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

        # Critic networks (local and target)
        self.critic_local = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

        # Replay memory buffer
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Exploration noise
        self.exploration_noise = OUNoise(action_size, seed)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.exploration_noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            self.learn()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples."""
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Convert all states and actions arrays to Tensors
        states = torch.from_numpy(states).float().to(device)
        actions = torch.from_numpy(actions).float().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target(next_states)
        y = rewards + GAMMA * self.critic_target(next_states, next_actions) * (1 - dones)

        # Compute critic loss and update critic network
        critic_loss = F.mse_loss(y, self.critic_local(states, actions))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss and update actor network
        cur_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, cur_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_network()

    def update_target_network(self):
        """Soft update model parameters."""
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process for generating correlated noise for exploration in DDPG."""

    def __init__(self, size, seed, mu=0.0, theta=0.1, sigma=0.2, sigma_min=0.1, sigma_decay=0.99):
        """
        Initialize parameters and the noise process.
        
        Parameters:
        - size (int): Dimension of the noise
        - seed (int): Random seed for reproducibility
        - mu (float): Long-running mean (default: 0.0)
        - theta (float): Speed of mean reversion (default: 0.1)
        - sigma (float): Volatility parameter (default: 0.2)
        - sigma_min (float): Minimum value for sigma after decay (default: 0.1)
        - sigma_decay (float): Decay rate for sigma per episode (default: 0.99)
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state to the mean (mu) and apply sigma decay."""
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        """Update internal state using Ornstein-Uhlenbeck process and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
