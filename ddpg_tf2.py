"""


"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
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
        self.min_action = env.action_space.low[0]
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

    def load_models(self):
        print("...... loadning models .......")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        """
        - Observation of current state of env.
        - Evaluate => Training vs Testing of agent. Add noise in training.
        """
        #state = tf.convert_to_tensor([observation], dtype=tf.float32) # Add extra dim as "batch dim"
        try:
            states = np.reshape(observation[0],(3,1))
        except ValueError:
            states = np.reshape(observation,(3,1))
        actions = self.actor(states)

        if not evaluate: # add normal noise
            # prolem, could add something to get outside of bound of environment
            actions += tf.random.normal(shape=[self.n_actions],
                                         mean=0.0, stddev=self.noise)
            # Clip it, to avoid going outside env.
            actions = tf.clip_by_value(actions, self.min_action, self.max_action)

            return actions[0] # return zeroth element of tensor => a numpy array
        
    def learn(self):
        """ The bread and butter
        Dilemma: what if we haven't filled a batch size of replay buffer yet?
            1: Wait until we fill up memory
            2: Do batch size number of random actions -> Then learn
        """
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
              self.memory.sample_buffer(self.batch_size)
        # Convert to tensors
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)

        # 1: Critic loss, using the target critic
        with tf.GradientTape() as tape:
            """
            Gradient tape: stick in update rule into gradient decent.
                - tf is made for deep learning, but DRL requires a bit more involved loss.
            """
            target_actions = self.target_actor(states_) # of the next state, what should we do?
            critic_value_ = tf.squeeze(self.target_critic(
                states_, target_actions), 1) # Critic value of new state
            # The actual state and action agent took during episode.
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            # The (1-done) will be 0 if the episode is over, as there are no resulting state.
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
        # Calculate and apply gradients
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        # 2: actor loss, using target actor
        # Essentially same approach, using context manager tape.
        with tf.GradientTape() as tape:
            # Actions according to current actor, not based on what it had in the memory.
            new_policy_actions = self.actor(states)
            # Loss using gradient -ascent-.
            # In Policy Gradient methods (PG) we want to maximize score over time => Ascent.
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        # Calculate and apply gradients. 
        actor_networ_gradient = tape.gradient(actor_loss,
                                              self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_networ_gradient, self.actor.trainable_variables
        ))

        # Soft update for target.
        self.update_network_parameters()