# ------------------------------------------------------------------------------
#   @ Description
#       Training Data Generation
#-------------------------------------------------------------------------------


import numpy as np
from config import configuration


class TrainingData(object):

    """
        Definition of Training Data Structure: X, U, actual cost
    """
    def __init__(self, X=None, U=None, cost=None):
        self.X = X              # (N, T+2, state_dim)
        self.U = U              # (N, T+1, action_dim)
        self.cost = cost


## generate samples with specified local controller

class Sampler(object):

    """
        Generation of Training Data using current local control law
    """

    def __init__(self, env, N, T, state_dim, action_dim):

        self.env = env
        self.N = N
        self.T = T  # t=0, 1, ..., T
        self.state_dim = state_dim
        self.action_dim = action_dim

        # get action bound
        self.action_bound = env.action_space.high[0]

    ## generate samples with specified local Gaussian controller
    def generate(self, x0, local_controller, cost_param, goal_state):

        X = np.zeros((self.N, self.T+2, self.state_dim))   # X=(x0, x1, ..., xT, xT+1)
        U = np.zeros((self.N, self.T+1, self.action_dim))  # U = (u0, u1, ..., uT)

        Kt = local_controller.Kt
        kt = local_controller.kt
        St = local_controller.St

        state = np.zeros(self.state_dim)

        for traj_no in range(self.N):

            # reset the environment to the same initial conditions
            bad_init = True
            while bad_init:
                state = self.env.reset()  # shape of observation from gym (3,)
                x0err = state - x0
                if np.sqrt(x0err.T.dot(x0err)) < 0.1:  # x0=(state_dim,)
                    bad_init = False

            for t in range(self.T+1):
                # visualize the environment
                if configuration['render_ok']:
                    self.env.render()

                # compute action
                mean_action = Kt[t, :, :].dot(state) + kt[t, :]
                action = np.random.multivariate_normal(mean=mean_action, cov=St[t, :, :])
                action = np.clip(action, -self.action_bound, self.action_bound)

                # collect trajectory
                X[traj_no, t, :] = state
                U[traj_no, t, :] = action


                # observe next_state, shape of output of gym (state_dim,)
                state, _, _, _ = self.env.step(action)

                if t == self.T:
                    # collect trajectory
                    X[traj_no, t+1, :] = state

        cost = self.actual_cost(X, U, cost_param, goal_state)

        return TrainingData(X, U, cost)


    ## evaluate cost with real data
    def actual_cost(self, X, U, cost_param, goal_state):

        cost = 0
        for traj_no in range(self.N):
            for t in range(self.T+1):
                x = X[traj_no, t, :]
                u = U[traj_no, t, :]
                cost = cost + (x-goal_state).T.dot(cost_param['wx']).dot(x-goal_state) + \
                       u.T.dot(cost_param['wu']).dot(u)

        cost = cost / self.N

        return cost
