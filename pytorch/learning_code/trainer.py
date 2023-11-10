# This file handles the actual training between the environment to our networks
from network_parameters import Hyperparameters

from networks import DQN

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as AdamW
import numpy as np

class TrainingAids(Hyperparameters):
    def __init__(self):
        super().__init__()

    def make_DQN_networks(self, nn_args):
        return {'q_eval':DQN(**nn_args), 'q_next':DQN(**nn_args)}

    def DQN_choose_action(self, observation, networks):
        state = T.tensor(observation, dtype = T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(state)
        return T.argmax(action_values).item()

    def learn_DQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.long))

        q_pred = networks['q_eval'](states)[indices, actions.type(T.LongTensor)]

        q_next = networks['q_next'](states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = networks['q_eval'].loss(q_target, q_pred).to(networks['q_eval'].device)
        loss.backward()

        networks['q_eval'].optimizer.step()
        networks['learn_step_counter'] += 1

        self.decrement_epsilon()

        return loss.item()

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon-self.eps_dec, self.eps_min)

    def store_transition(self, s, a, r, s_, d, networks):
        networks['replay'].store_transition(s, a, r, s_, d)

    def sample_memory(self, networks):
        states, actions, rewards, states_, dones = networks['replay'].sample_buffer(self.batch_size)
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            device = networks['q_eval'].device
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG', 'TD3'}:
            device = networks['actor'].device
        states = T.tensor(states, dtype=T.float32).to(device)
        actions = T.tensor(actions, dtype=T.float32).to(device)
        rewards = T.tensor(rewards, dtype=T.float32).to(device)
        states_ = T.tensor(states_, dtype=T.float32).to(device)
        dones = T.tensor(dones).to(device)

        return states, actions, rewards, states_, dones