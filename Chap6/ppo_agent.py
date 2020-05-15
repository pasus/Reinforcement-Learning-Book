# PPO with GAE Agent for training and evaluation

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

from ppo_actor import Actor
from ppo_critic import Critic

class PPOagent(object):

    def __init__(self, env):

        self.sess = tf.Session()
        K.set_session(self.sess)

        # hyperparameters
        self.GAMMA = 0.95
        self.GAE_LAMBDA = 0.9 #0.8
        self.BATCH_SIZE = 64
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.RATIO_CLIPPING = 0.2
        self.EPOCHS = 10

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        # create actor and critic networks
        self.actor = Actor(self.sess, self.state_dim, self.action_dim, self.action_bound,
                           self.ACTOR_LEARNING_RATE, self.RATIO_CLIPPING)
        self.critic = Critic(self.state_dim, self.action_dim, self.CRITIC_LEARNING_RATE)

        # initialize for later gradient calculation
        self.sess.run(tf.global_variables_initializer())  #<-- no problem without it

        # save the results
        self.save_epi_reward = []


    ## computing Advantages and targets: y_k = r_k + gamma*V(s_k+1), A(s_k, a_k)= y_k - V(s_k)
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.GAMMA * forward_val - v_values[k]
            gae_cumulative = self.GAMMA * self.GAE_LAMBDA * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets


    ## train the agent
    def train(self, max_episode_num):

        # initialize batch
        states, actions, rewards = [], [], []
        log_old_policy_pdfs = []

        for ep in range(int(max_episode_num)):

            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset() # shape of state from gym (3,)

            while not done:

                # visualize the environment
                #self.env.render()
                # compute mu and std of old policy and pick an action (shape of gym action = (action_dim,) )
                mu_old, std_old, action = self.actor.get_policy_action(state)
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # compute log old policy pdf
                var_old = std_old ** 2
                log_old_policy_pdf = -0.5 * (action - mu_old) ** 2 / var_old - 0.5 * np.log(var_old * 2 * np.pi)
                log_old_policy_pdf = np.sum(log_old_policy_pdf)
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)

                # append to the batch
                states.append(state)
                actions.append(action)
                rewards.append((reward+8)/8) # <-- normalization
                log_old_policy_pdfs.append(log_old_policy_pdf)

                # if batch is full, start to train networks on batch
                if len(states) == self.BATCH_SIZE:

                    # compute gae and TD targets
                    next_v_value = self.critic.predict([next_state])
                    v_values = self.critic.predict(states)
                    gaes, y_i = self.gae_target(rewards, v_values, next_v_value, done)

                    # update the networks
                    for _ in range(self.EPOCHS):

                        # train
                        self.actor.train(log_old_policy_pdfs, states, actions, gaes)
                        self.critic.train_on_batch(states, y_i)

                    # clear the batch
                    states, actions, rewards = [], [], []
                    log_old_policy_pdfs = []

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

