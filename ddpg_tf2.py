"""


"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optmizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=65, noise=0.1) -> None:
        """
        params:
        - alpha: lr for the actor
        - beta: lr for the critic, higher as policy approximation is more sensitive than the value one.
        - env: to add noice 
        - gamma: discount
        - n_actions: action dimensions.
        - max_size: of replay buffer
        - tau: soft update rule from paper.
        - fc1
        """
        #
        self.gamma=gamma
        self.tau=tau
        #
        self.memory=ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions=n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.max_action = env.action_space.low[0]
        #
        self.actor = ActorNetwork(n_actions=n_actions, name="actor")
        self.critic = CriticNetwork(name="critic")
        self.target_actor = ActorNetwork(n_actions=n_actions, name="target_actor")
        self.target_critic = CriticNetwork(name="target_critic")

        # Compile networks
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        # Arent actually going to do gradient decent on these networks, only soft updates.
        # Have to compile them anyways. TF2 stuff.
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        # Do the first initial hard copy of the networks
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        """
        Implementing hardcopy as well as soft copy from paper.
        """
        # Actor
        if tau is None:
            tau=self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        # Critic
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        # Just a clean transition function
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print("...... saving models ......")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_model(self):
        print("...... loadning models .......")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self):
        ...