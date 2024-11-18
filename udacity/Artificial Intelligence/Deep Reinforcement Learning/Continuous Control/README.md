# Project Title

This project focuses on training an agent in a simulated environment using Deep Deterministic Policy Gradient (DDPG) algorithm.

## Project Details

### Environment Details

- **State Space**: The state space consists of `state_size` continuous variables, represented by a vector. Each value in the vector is a feature of the state observed from the environment.
- **Action Space**: The action space consists of `action_size` continuous variables. Each variable represents a possible action the agent can undertake in the environment.
- **Solving the Environment**: The environment is considered solved when the agent achieves an average score of 30.0 over 100 consecutive episodes.

## Getting Started

To get started with this project, follow these steps to set up the necessary environment and dependencies:

1. **Install Dependencies**:
    Ensure you have Python 3.8.10 installed. You can create a virtual environment and install dependencies using `pip` as follows:
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

2. **Additional Files**:
    - Download the Unity environment file and place it in the root of your project directory.

## Instructions

To train the agent using the provided code, follow these steps:

1. **Launch the Training**:
    Run the training script to start training the agent:
    ```shell
    python train.py
    ```

    This script initializes the environment and the agent, then proceeds to train the agent for a specified number of episodes. The models will be saved to `checkpoint_actor.pth` and `checkpoint_critic.pth` once the environment is solved.

2. **Monitor Training**:
    The script will print the average score over the last 100 episodes to the console. You can observe the agent’s performance improvement over time.

3. **Modify Parameters**:
    You can also modify the hyperparameters (like number of episodes, maximum timesteps per episode, learning rates, etc.) in the `train.py` file to experiment with different settings.

### Sample Command:
```shell
python train.py
```

This command will initiate the training process for the agent.

## Additional Resources

For more information on creating effective README files and using Markdown, refer to the following resources:
- [Creating and Managing a README](https://guides.github.com/features/wikis/#creating-and-editing-pages)
- [Mastering Markdown](https://guides.github.com/features/mastering-markdown/)

---

Happy Training!