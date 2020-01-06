# DDPG Critic

from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam

import tensorflow as tf


class Critic(object):
    """
        Critic Network for DDPG: Q function approximator
    """
    def __init__(self, sess, state_dim, action_dim, tau, learning_rate):
        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate

        # create critic and target critic network
        self.model, self.states, self.actions = self.build_network()
        self.target_model, _, _ = self.build_network()

        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        self.target_model.compile(optimizer=Adam(self.learning_rate), loss='mse')

        # compute dq_da to feed to the actor
        self.q_grads = tf.gradients(self.model.output, self.actions)

    ## critic network
    def build_network(self):
        state_input = Input((self.state_dim,))
        action_input = Input((self.action_dim,))
        x1 = Dense(64, activation='relu')(state_input)
        x2 = Dense(32, activation='linear')(x1)
        #a1 = Dense(1, activation='linear')(action_input)
        a1 = Dense(32, activation='linear')(action_input)
        h2 = concatenate([x2, a1], axis=-1)
        #h2 = Add()([x2, a1])
        h3 = Dense(16, activation='relu')(h2)
        q_output = Dense(1, activation='linear')(h3)
        model = Model([state_input, action_input], q_output)
        model.summary()
        return model, state_input, action_input


    ## q-value prediction of target critic
    def target_predict(self, inp):
        return self.target_model.predict(inp)


    ## transfer critic weights to target critic with a aau
    def update_target_network(self):
        phi = self.model.get_weights()
        target_phi = self.target_model.get_weights()
        for i in range(len(phi)):
            target_phi[i] = self.tau * phi[i] + (1 - self.tau) * target_phi[i]
        self.target_model.set_weights(target_phi)


    ## gradient of q-values wrt actions
    def dq_da(self, states, actions):
        return self.sess.run(self.q_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })

    ## single gradient update on a single batch data
    def train_on_batch(self, states, actions, td_targets):
        return self.model.train_on_batch([states, actions], td_targets)

    ## save critic weights
    def save_weights(self, path):
        self.model.save_weights(path)


    ## load critic wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5')