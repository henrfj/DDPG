""" To store [s,a,r,s+1] Touples"""

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions) -> None:
        """
        params:
        - n_actions : is not number of possible actions (as it is discrete), but components of an actions (dimensions).
        """
        self.mem_size = max_size # overwrite old with new one
        self.mem_cntr = 0
        #
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        #
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        #
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, new_state, done):
        """
        """
        # Position of first available memory index
        index = self.mem_cntr % self.mem_size
        # Save tranition values
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        # Flags as value of terminal state is zero. Reset the eposiode. Is state terminal?
        self.terminal_memory[index] = done 

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Need to sample from the buffer, while also making sure that the we have enough data to sample from.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False) # prevents double sample
        
        states = self.state_memory[batch]
        states_  = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones