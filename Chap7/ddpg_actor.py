# DDPG Actor

import numpy as np

from keras.models import Model
from keras.layers import Dense, Input, Lambda

import tensorflow as tf

class Actor(object):
    """
        Actor Network for DDPG
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, tau, learning_rate):
        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau
        self.learning_rate = learning_rate

        self.model, self.theta, self.states = self.build_network()
        self.target_model, self.target_theta, _ = self.build_network()

        self.dq_da_ph = tf.placeholder(tf.float32, [None, self.action_dim])
        self.dj_dtheta = tf.gradients(self.model.output, self.theta, -self.dq_da_ph)
        grads = zip(self.dj_dtheta, self.theta)
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

    ## actor network
    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out = Dense(self.action_dim, activation='tanh')(h3)

        # Scale output to [-action_bound, action_bound]
        action_output = Lambda(lambda x: x*self.action_bound)(out)
        model = Model(state_input, action_output)
        model.summary()
        return model, model.trainable_weights, state_input


    ## actor prediction
    def predict(self, state):
		# type of action in env is numpy array
        return self.model.predict(np.reshape(state, [1, self.state_dim]))[0]


    ## target actor prediction
    def target_predict(self, state):
        return self.target_model.predict(state)


    ## transfer actor weights to target actor with a aau
    def update_target_network(self):
        theta, target_theta = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(theta)):
            target_theta[i] = self.tau * theta[i] + (1 - self.tau) * target_theta[i]
        self.target_model.set_weights(target_theta)


    ## train the actor network
    def train(self, states, dq_das):
        self.sess.run(self.actor_optimizer, feed_dict={
            self.states: states,
            self.dq_da_ph: dq_das
        })


    ## save actor weights
    def save_weights(self, path):
        self.model.save_weights(path)


    ## load actor wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_actor.h5')
