import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
        """Initialize parameters and build model."""
        super(ActorNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)
        self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic (Q-value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
        """Initialize parameters and build model."""
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1 + action_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
