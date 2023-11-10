import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, num_observations, num_actions, action_type = None, use_intention = False):
        self.memory_size = max_size # How large our memory buffer is
        self.use_intention = use_intention # Simple bool so we know what type of memory to use
        self.memory_counter = 0 # Memory counter to know what space in mem we are in
        self.action_type = action_type
        self.state_memory = np.zeros((self.memory_size, num_observations), dtype = np.single)
        self.new_state_memory = np.zeros((self.memory_size, num_observations), dtype = np.single)
        if use_intention:
            self.action_memory = np.zeros((self.memory_size), dtype=np.single)
        else:
            if self.action_type == 'Discrete':
                self.action_memory = np.zeros((self.memory_size), dtype = np.intc)
            elif self.action_type == 'Continuous':
                self.action_memory = np.zeros((self.memory_size, num_actions), dtype = np.single)
            else:
                raise Exception('Invalid Action Type:' + action_type)
        self.reward_memory = np.zeros((self.memory_size), dtype = np.single)
        self.terminal_memory = np.zeros((self.memory_size), dtype = np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        mem_index = self.memory_counter % self.mem_size
        self.state_memory[mem_index] = state
        if self.use_intention:
            self.action_memory[mem_index] = action
        else:
            if self.action_type == 'Discrete':
                self.action_memory[mem_index] = action[0]
            elif self.action_type == 'Continuous':
                self.action_memory[mem_index] = action[1][0:2]
        self.reward_memory[mem_index] = reward
        self.new_state_memory[mem_index] = state_
        self.terminal_memory[mem_index] = done
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        num_done = self.terminal_memory[batch]

        return states, actions, rewards, next_states, num_done