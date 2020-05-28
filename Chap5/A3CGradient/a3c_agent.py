# A3C Global agent and local agents for training and evaluation
# grad parallelism

import gym

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt

import threading
import multiprocessing

from a3c_actor import Global_Actor, Worker_Actor
from a3c_critic import Global_Critic, Worker_Critic

# shared global parameters across all workers
global_episode_count = 0
global_step = 0
global_episode_reward = []  # save the results


class A3Cagent(object):

    """
        Global network
    """
    def __init__(self, env_name):

        self.sess = tf.Session()
        K.set_session(self.sess)

        # training environment
        self.env_name = env_name
        self.WORKERS_NUM = multiprocessing.cpu_count() #4

        # get state dimension
        env = gym.make(self.env_name)
        state_dim = env.observation_space.shape[0]
        # get action dimension
        action_dim = env.action_space.shape[0]
        # get action bound
        action_bound = env.action_space.high[0]

        # create global actor and critic networks
        self.global_actor = Global_Actor(state_dim, action_dim, action_bound)
        self.global_critic = Global_Critic(state_dim)

        # initialize for later gradient calculation
        #self.sess.run(tf.global_variables_initializer())

    def train(self, max_episode_num):

        workers = []

        # create worker
        for i in range(self.WORKERS_NUM):
            worker_name = 'worker%i' % i
            workers.append(A3Cworker(worker_name, self.env_name, self.sess, self.global_actor,
                                     self.global_critic, max_episode_num))

        # initialize for later gradient calculation
        self.sess.run(tf.global_variables_initializer())


        # create worker (multi-agents) and do parallel training
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


        np.savetxt('./save_weights/pendulum_epi_reward.txt', global_episode_reward)
        print(global_episode_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(global_episode_reward)
        plt.show()


class A3Cworker(threading.Thread):

    """
        local agent network (worker)
    """
    def __init__(self, worker_name, env_name, sess, global_actor, global_critic, max_episode_num):
        threading.Thread.__init__(self)

        #self.lock = threading.Lock()

        # hyperparameters
        self.GAMMA = 0.95
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.ENTROPY_BETA = 0.01
        self.t_MAX = 4 # t-step prediction

        self.max_episode_num = max_episode_num

        # environment
        self.env = gym.make(env_name)
        self.worker_name = worker_name
        self.sess = sess

        # global network sharing
        self.global_actor = global_actor
        self.global_critic = global_critic



        # get state dimension
        self.state_dim = self.env.observation_space.shape[0]
        # get action dimension
        self.action_dim = self.env.action_space.shape[0]
        # get action bound
        self.action_bound = self.env.action_space.high[0]

        # create local actor and critic networks
        self.worker_actor = Worker_Actor(self.sess, self.state_dim, self.action_dim, self.action_bound,
                                         self.ACTOR_LEARNING_RATE, self.ENTROPY_BETA, self.global_actor)
        self.worker_critic = Worker_Critic(self.sess, self.state_dim, self.action_dim,
                                           self.CRITIC_LEARNING_RATE, self.global_critic)

        # initial transfer global network parameters to worker network parameters
        self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
        self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

    ## computing Advantages and targets: y_k = r_k + gamma*V(s_k+1), A(s_k, a_k)= y_k - V(s_k)
    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.GAMMA * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets


    # train each worker
    def run(self):

        global global_episode_count, global_step
        global global_episode_reward  # total episode across all workers

        print(self.worker_name, "starts ---")

        while global_episode_count <= int(self.max_episode_num):

            # initialize batch
            states, actions, rewards = [], [], []

            # reset episode
            step, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset() # shape of state from gym (3,)

            while not done:

                # visualize the environment
                #self.env.render()
                # pick an action (shape of gym action = (action_dim,) )
                action = self.worker_actor.get_action(state)
                # clip continuous action to be within action_bound
                action = np.clip(action, -self.action_bound, self.action_bound)
                # observe reward, new_state, shape of output of gym (state_dim,)
                next_state, reward, done, _ = self.env.step(action)

                # append to the batch
                states.append(state)
                actions.append(action)
                rewards.append((reward+8)/8) # <-- normalization

                # if batch is full or episode ends, start to train worker on batch
                if len(states) == self.t_MAX or done:

                    # compute n-step TD target and advantage prediction
                    next_v_value = self.worker_critic.predict([next_state])
                    n_step_td_targets = self.n_step_td_target(rewards, next_v_value, done)
                    v_values = self.worker_critic.predict(states)
                    advantages = n_step_td_targets - v_values


                    #with self.lock:
                    # update global critic
                    self.worker_critic.train(states, n_step_td_targets)
                    # update global actor
                    self.worker_actor.train(states, actions, advantages)

                    # transfer global network parameters to worker network parameters
                    self.worker_actor.model.set_weights(self.global_actor.model.get_weights())
                    self.worker_critic.model.set_weights(self.global_critic.model.get_weights())

                    # clear the batch
                    states, actions, rewards = [], [], []

                    # update global step
                    global_step += 1

                # update state and step
                state = next_state
                step += 1
                episode_reward += reward

            # update global episode count
            global_episode_count += 1
            ## display rewards every episode
            print('Worker name:', self.worker_name, ', Episode: ', global_episode_count,
                  ', Step: ', step, ', Reward: ', episode_reward)

            global_episode_reward.append(episode_reward)

            ## save weights every episode
            if global_episode_count % 10 == 0:
                self.global_actor.save_weights("./save_weights/pendulum_actor.h5")
                self.global_critic.save_weights("./save_weights/pendulum_critic.h5")

