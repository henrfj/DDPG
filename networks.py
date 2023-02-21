import os # File path joining operation =>  model checkpoint
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512,
                 name="critic", chkpt_dir="tmp/ddpg") -> None:
        super(CriticNetwork, self).__init__()
        #
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        #
        self.model_name = name # used to distinguish target network from regular networks.
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5') # Keras model file type
        #
        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None) 
        """
        NB! In original paper they use batch normalization. Not needed but adds just a little.
            - They also tweak layer initialization, but not needed.
        """

    def call(self, state, action):
        # Forward pass of critic
        action_value = self.fc1(tf.concat[state, action], axis=1) #0'th axis is the batch
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q
    

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=512, fc2_dims=512, n_actions=2,
                  name="actor", chkpt_dir="tmp/ddpg") -> None:
        super(ActorNetwork, self).__init__()
        #
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        #
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.model_name+'_ddpg.h5') # Keras model file type
        #
        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(self.n_actions, activation="tanh")
        """
        Want a bounded activation function between -1, +1, and then multiplying if we need more of environment.
        """ 

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        # Can either multiply here, or in "choose action" function in agent class.
        mu = self.mu(prob) 
        return mu