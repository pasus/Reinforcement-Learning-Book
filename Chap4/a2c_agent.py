# A2C Agent for training and evaluation

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from a2c_actor import Actor
from a2c_critic import Critic

class A2Cagent(object):

    def __init__(self, env):

        self.sess = tf.Session()
        K.set_session(self.sess)

        # hyperparameters
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        # create actor and critic networks
        self.actor = Actor(self.sess, self.state_dim, self.action_dim, self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        # initialize for later gradient calculation
        self.sess.run(tf.global_variables_initializer())  #<-- no problem without it

        # save the results
        self.save_epi_reward = []


    ## computing Advantages and targets: y_k = r_k + gamma*V(s_k+1), A(s_k, a_k)= y_k - V(s_k)
    def advantage_td_target(self, reward, v_value, next_v_value, done):
        if done:
                y_k = reward
                advantage = y_k - v_value
        else:
                y_k = reward + self.GAMMA * next_v_value
                advantage = y_k - v_value
        return advantage, y_k


    ## train the agent
    def train(self, max_episode_num):

        for ep in range(int(max_episode_num)):

            # initialize batch
            states, actions, td_targets, advantages = [], [], [], []
            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset() # shape of state from gym (3,)

            while not done:

                # visualize the environment
                #self.env.render()
                # pick an action (shape of gym action = (action_dim,) )
                action = self.actor.get_action(state)
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)
                # compute next v_value
                v_value = self.critic.predict(state)
                next_v_value = self.critic.predict(next_state)
                # compute advantage and TD target
                train_reward = (reward+8)/8  # <-- normalization
                advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)

                # append to the batch
                states.append(state)
                actions.append(action)
                td_targets.append(y_i)
                advantages.append(advantage)

                # if batch is full, start to train networks on batch
                if len(states) == self.BATCH_SIZE:
                    # train critic
                    self.critic.train_on_batch(states, td_targets)
                    # train actor
                    self.actor.train(states, actions, advantages)

                    # clear the batch
                    states, actions, td_targets, advantages = [], [], [], []

                # update current state
                state = next_state
                episode_reward += reward
                time += 1


            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)


            ## save weights every episode
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/pendulum_actor.h5")
                self.critic.save_weights("./save_weights/pendulum_critic.h5")

        np.savetxt('./save_weights/pendulum_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

