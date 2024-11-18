# Collaboration and Competition

## Project Details

This project utilizes the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train two agents in the Unity Tennis environment. The environment is considered solved when the agents achieve an average score of 0.5 over 100 consecutive episodes.

### Requirements:
- Python 3.9.6
- Dependencies listed below

## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/nvnhann/Collaboration-and-Competition
    cd Collaboration-and-Competition
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the Unity environment for your operating system:
   - [Linux](https://drive.google.com/...)
   - [Mac](https://drive.google.com/...)
   - [Windows](https://drive.google.com/...)

4. Extract the environment and place it in the `data` directory.

### Instructions

To train the agent, run the following command:

```bash
python train.py --env_file "/data/Tennis_Linux_NoVis/Tennis"
```

### File Structure

- `model.py`: Defines the Actor and Critic models.
- `maddpg.py`: Contains the MADDPG agent and the replay buffer.
- `train.py`: Script to train the MADDPG agent in the Unity Tennis environment.

### Running the Training

1. Ensure that the Unity environment file path is correctly specified in the command.
2. Execute the training script:
    ```bash
    python train.py --env_file "/data/Tennis_Linux_NoVis/Tennis"
    ```
3. The training progress and average score will be printed to the console.
4. A plot of the scores will be saved as `scores_plot.png` after training.

## Additional Resources

For more information on creating effective README files and Markdown formatting, refer to the following resources:
- [Creating a README](https://example.com)
- [Markdown Guide](https://example.com)

---