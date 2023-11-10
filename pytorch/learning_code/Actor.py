from trainer import TrainingAids
from replay_buffer import ReplayBuffer
import torch as T
import numpy as np

class Actor(TrainingAids):
    def __init__(self, n_obs, n_actions, options_per_action, n_agents, n_chars, meta_param_size,
                 intention = False, seq_len=5):

        super().__init__()

        self.n_obs = n_obs
        self.n_actions = n_actions
        self.options_per_action = options_per_action

        self.n_chars = n_chars
        self.n_agents = n_agents
        self.meta_param_size = meta_param_size

        self.action_space = [i for i in range(self.options_per_action**self.n_actions)]
        self.failure_action_code = len(self.action_space)

        self.intention = intention
        self.seq_len = seq_len

        self.network_input_size = self.n_obs
        if self.intention:
            self.network_input_size += 1
        if self.intention_neighbors:
            self.intention_network_input = 2+2*2  # [own prox, neighbors prox, own prev gsp, neighbors prev gsp]
            if self.attention_intention:
                self.attention_observation = [[0 for _ in range(self.intention_network_input)] for _ in range(self.seq_len)]
            #TODO write logic for recurrent and attention
        else:
            self.intention_network_input = self.n_agents*self.n_chars
            if self.attention_intention:
                self.attention_observation = [[0 for _ in range(2+self.n_agents*self.n_chars)] for _ in range(self.seq_len)]
            elif self.recurrent_intention:
                self.recurrent_intention_network_input = self.intention_network_input + self.meta_param_size

    def build_networks(self, learning_scheme):
        if learning_scheme == 'None':
            self.networks = {'learning_scheme': '', 'learn_step_counter': 0}
        if learning_scheme == 'DQN':
            nn_args = {'id': self.id, 'lr': self.lr, 'num_actions': self.n_actions,
                       'observation_size': self.network_input_size,
                       'num_ops_per_action': self.options_per_action}
            self.networks = self.build_DQN(nn_args)
            self.networks['learning_scheme'] = 'DQN'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete')
            self.networks['learn_step_counter'] = 0

    def build_DQN(self, nn_args):
        return self.make_DQN_networks(nn_args)

    def save_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].save_checkpoint(path)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau

    def replace_target_network(self):
        if self.networks['learn_step_counter'] % self.replace_target_ctr==0:
            self.networks['q_next'].load_state_dict(self.networks['q_eval'].state_dict())

    def choose_action(self, observation, networks, edge_index = None, test=False):
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            if test or np.random.random()>self.epsilon:
                actions = self.DQN_DDQN_choose_action(observation, networks)
            else:
                actions = np.random.choice(self.action_space)
            return actions

    def learn(self, edge_index=None):
        if self.networks['replay'].mem_ctr < (self.n_agents*self.batch_size + self.batch_size):
                return

        if self.networks['learning_scheme'] == 'DQN':
            self.replace_target_network()

            return self.learn_DQN(self.networks)

    def store_agent_transition(self, s, a, r, s_, d):
        self.store_transition(s, a, r, s_, d, self.networks)

    def save_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].save_checkpoint(path)

    def load_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].load_checkpoint(path)

