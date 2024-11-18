import argparse
from collections import deque
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from maddpg import MADDPG

LEN_DEQUE = 100


def train(agent: MADDPG, env: UnityEnvironment, n_episodes: int, print_every: int, threshold: float, brain_name: str,
          max_len_deque: int = LEN_DEQUE) -> List[float]:
    """Train the agent using MADDPG algorithm.

    Args:
        agent (MADDPG): The multi-agent DDPG instance.
        env (UnityEnvironment): The Unity environment.
        n_episodes (int): Number of training episodes.
        print_every (int): Interval for printing average scores.
        threshold (float): The score threshold to consider the environment solved.
        brain_name (str): The brain name of the environment.
        max_len_deque (int, optional): The maximum length of the deque to store recent scores. Defaults to LEN_DEQUE.

    Returns:
        List[float]: Scores from each episode.
    """
    scores_deque = deque(maxlen=max_len_deque)
    scores = []

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        agent.reset()

        while True:  # The task is episodic
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break

        episode_score = np.max(score)  # Single score for the episode just played
        scores.append(episode_score)
        scores_deque.append(episode_score)

        if len(scores_deque) == max_len_deque and np.mean(scores_deque) >= threshold:
            print(f"Environment was solved at episode {i_episode - max_len_deque}")
            agent.save_weights()
            return scores

        if i_episode % print_every == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.3f}')

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', '-e', default='', type=str, help="Path to Unity environment file")
    parser.add_argument('--n_episodes', '-n', default=4000, type=int, help="Number of episodes to train the agent")
    parser.add_argument('--print_every', default=50, type=int, help="Interval for printing the average score")
    args = parser.parse_args()

    # Instantiate the environment
    env = UnityEnvironment(file_name=args.env_file)

    # Get the default brain
    brain_name = env.brain_names[0]

    # Instantiate the agent
    state_size_per_agent = 8
    num_stacked_obs = 3
    state_size = state_size_per_agent * num_stacked_obs
    action_size_per_agent = 2
    agent = MADDPG(state_size=state_size, action_size=action_size_per_agent, random_seed=0)

    # Train with MADDPG algorithm
    threshold = 0.5
    scores = train(agent, env, args.n_episodes, args.print_every, threshold, brain_name)

    # Plot scores
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores_plot.png')
    plt.show()

    # Close env
    env.close()


if __name__ == "__main__":
    main()
