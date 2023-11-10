import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math

class DQN(nn.Module):
    def __init__(self, lr, n_actions, observation_size,
                 fc1_dims = 64, fc2_dims = 128, name = 'DQN'):
        super().__init__()

        self.name = name

        output_dims = n_actions

        self.fc1 = nn.Linear(observation_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_dims)

        self.optimizer = optim.AdamW(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x1 = F.relu(self.fc2(x))
        actions = self.fc3(x1)

        return actions

    def save_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path, intention=False):
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))