import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment
from agent import Agent

# Change this path to the location of your Unity environment file
env = UnityEnvironment(file_name="/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64")

# Load the brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Hyperparameters
n_episodes = 2000  # Total number of episodes
max_t = 1000  # Maximum number of timesteps per episode
print_every = 100  # Print frequency


def train_ddpg():
    """Train the agent using DDPG algorithm."""

    # Initialize the agent
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    agent = Agent(state_size=state_size, action_size=action_size, seed=0)

    # Tracking scores
    scores = []
    scores_window = deque(maxlen=print_every)

    # Training loop
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
