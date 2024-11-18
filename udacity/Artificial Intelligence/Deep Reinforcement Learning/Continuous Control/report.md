# Training Agent Report

## Learning Algorithm

In this project, we used the Deep Deterministic Policy Gradient (DDPG) algorithm to train an agent to interact with the environment. The DDPG algorithm is a model-free, off-policy algorithm used for learning in environments with continuous action spaces.

### Hyperparameters

- Number of Episodes (`n_episodes`): 2000
- Maximum Timesteps per Episode (`max_t`): 1000
- Printing Frequency (`print_every`): 100
- Actor and Critic Learning Rates (`LR`): 3e-4
- Discount Factor (`GAMMA`): 0.99
- Batch Size (`BATCH_SIZE`): 64
- Soft Update Parameter (`TAU`): 1e-3
- Replay Buffer Size (`BUFFER_SIZE`): 1,000,000

### Model Architectures

#### Actor Network

The Actor network is responsible for selecting actions based on the state. It maps state variables to action variables using the following architecture:

```python
class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))
```

- **Input Layer**: Receives the state as input.
- **Hidden Layer 1**: A fully connected layer with 128 neurons, followed by ReLU activation and Batch Normalization.
- **Hidden Layer 2**: A fully connected layer with 128 neurons, followed by ReLU activation.
- **Output Layer**: A fully connected layer with the size of the action space, followed by Tanh activation.

#### Critic Network

The Critic network evaluates the actions taken by the Actor via mapping state-action pairs to Q-values using the following architecture:

```python
class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1 + action_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

- **Input Layer**: Receives the state as input.
- **Hidden Layer 1**: A fully connected layer with 128 neurons, followed by ReLU activation and Batch Normalization.
- **Hidden Layer 2**: A fully connected layer that concatenates the output of the previous state hidden layer with action input, followed by 128 neurons and ReLU activation.
- **Output Layer**: A fully connected layer with one neuron representing the Q-value.

### Agent

The Agent is responsible for interacting with the environment, storing experiences, and learning from them. It consists of Actor and Critic networks, an exploration noise process, and a replay buffer.

```python
class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.noise = OUNoise(action_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.actor_target(next_states)
        next_Q_values = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (gamma * next_Q_values * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
```

### Exploration Noise

To encourage exploration, the agent adds noise to the actions using the Ornstein-Uhlenbeck process.

```python
class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
    
    def reset(self):
        self.state = copy.copy(self.mu)
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state
```

### Replay Buffer

The Replay Buffer stores the agent's experiences and samples from them to make learning more efficient.

```python
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
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
```

## Training Loop

The following code snippet demonstrates the training process for the agent:

```python
import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from agent import Agent

# Change this path to the location of your Unity environment file
env = UnityEnvironment(file_name="path_to_unity_environment")

# Load the brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Hyperparameters
n_episodes = 2000
max_t = 1000
print_every = 100

def train_ddpg():
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    scores = []
    scores_window = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='')
        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
        if np.mean(scores_window) >= 30.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    env.close()
    return scores

if __name__ == "__main__":
    scores = train_ddpg()
```

## Plot of Rewards

Below is a plot of rewards for each training episode. Our goal is to achieve an average reward of at least +30 over consecutive 100 episodes.

![Reward Plot](training_scores.png)

**Number of Episodes to Solve Environment**: [number of episodes]

```python
import numpy as np
import matplotlib.pyplot as plt

# Scores obtained from each training episode
scores = [0.6299999859184027, ...]  # List of scores

# Plotting the rewards
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Plot of Rewards per Episode')
plt.show()

# Saving the plot as a .png file
plt.savefig('training_scores.png')
```

## Ideas for Future Work

1. **Tuning Hyperparameters**: Experiment with different hyperparameters, such as learning rates, batch size, and gamma values, to find the optimal configuration for the agent.
2. **Enhancing Network Architecture**: Experiment with more complex network architectures, such as using LSTM or GRU for handling sequential and time-dependent environments.
3. **Data Augmentation**: Adjust or extend the training data using augmentation techniques to help the network learn better.
4. **Parallel Learning**: Utilize parallel learning with a large number of agents to speed up the training process and improve stability.
5. **Hyperparameter Optimization**: Employ automated hyperparameter optimization tools, such as Bayesian Optimization or Grid Search, to find the best settings.

We hope that implementing these improvements will significantly boost the agent's performance in future training tasks.