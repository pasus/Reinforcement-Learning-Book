# PGagent for training and evaluation

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Dense, Input, Lambda

class PGagent(object):

    def __init__(self, env):
        self.env = env

        self.sess = tf.Session()
        K.set_session(self.sess)
        
        # hyperparameters
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.0001
        
        # get state dimension
        state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        self.std_bound = [1e-2, 1.0]  # std bound

        # save the results
        self.save_epi_reward = []

        self.model, theta, self.states = self.build_network(state_dim)

        self.actions = tf.placeholder(tf.float32, [None, self.action_dim])
        self.dc_rewards = tf.placeholder(tf.float32, [None, 1])

        # policy pdf
        mu_a, std_a = self.model.output
        log_policy_pdf = self.log_pdf(mu_a, std_a, self.actions)

        # loss function and its gradient
        loss_policy = log_policy_pdf * self.dc_rewards
        loss = tf.reduce_sum(-loss_policy)
        dj_dtheta = tf.gradients(loss, theta)
        grads = zip(dj_dtheta, theta)
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).apply_gradients(grads)

        # initialize for later gradient calculation
        self.sess.run(tf.global_variables_initializer())  #<-- no problem without it

    ## policy gradient network
    def build_network(self, state_dim):
        state_input = Input((state_dim,))
        h1 = Dense(64, activation='relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out_mu = Dense(self.action_dim, activation='tanh')(h3)
        std_output = Dense(self.action_dim, activation='softplus')(h3)

        # Scale output to [-self.action_bound, self.action_bound]
        mu_output = Lambda(lambda x: x*self.action_bound)(out_mu)
        model = Model(state_input, [mu_output, std_output])
        model.summary()
        return model, model.trainable_weights, state_input

    ## log policy pdf
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def get_action(self, state):
        mu_a, std_a = self.model.predict(np.array([state]))
        mu_a = mu_a[0]
        std_a = std_a[0]
        std_a = np.clip(std_a, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action

    ## actor prediction
    def predict(self, state):
        mu_a, _= self.model.predict(np.array([state]))
        return mu_a[0]

    ## train the agent
    def train(self, max_episode_num):
        for ep in range(int(max_episode_num)):
            # initialize batch
            states, actions, rewards = [], [], []
            # reset episode
            T, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset() # shape of state from gym (3,)

            while not done:
                # visualize the environment
#                 self.env.render()
                # pick an action (shape of gym action = (action_dim,) )
                action = self.get_action(state)
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)
                train_reward = (reward+8)/8  # <-- normalization

                # append to the batch
                states.append(state)
                actions.append(action)
                rewards.append(train_reward)

                # update current state
                state = next_state
                episode_reward += reward
                T += 1

            # compute discounted rewards(G^(m)_t) from the episode
            dc_rewards = [ sum([ self.GAMMA**(k-t) * rewards[k] for k in range(t, T) ]) for t in range(T) ]

            # train the agent
            self.sess.run(self.optimizer, feed_dict={
                self.states: states,
                self.actions: actions,
                self.dc_rewards: np.expand_dims(dc_rewards, axis=-1)
            })


            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', T, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            if ep % 100 == 0:
                self.save_weights("./save_weights/pendulum.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

    ## save weights
    def save_weights(self, path):
        self.model.save_weights(path)

    ## load wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum.h5')