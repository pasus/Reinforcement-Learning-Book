# ------------------------------------------------------------------------------
#
#       LQR_FLM for pendulum-v0
#       data collection type [X, U]
#       state = [x, y, theta_dot]
#
# ------------------------------------------------------------------------------

import copy
import numpy as np
import math

from sample_trajectory import TrainingData, Sampler
from linear_dynamics import DynamicsData, LocalDynamics
from gaussian_control import ControlData, LocalControl

from gmm.dynamics_prior_gmm import DynamicsPriorGMM

from config import configuration


class LQRFLMagent(object):

    def __init__(self, env):

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        # get action bound
        self.action_bound = env.action_space.high[0]

        # initialize GMM
        self.prior = DynamicsPriorGMM()

        # goal state
        goal_ang = 0 * np.pi / 180.0
        self.goal_state = np.array([math.cos(goal_ang), math.sin(goal_ang), 0])

        # initial condition
        i_ang = -45.0 * np.pi / 180.0
        self.init_state = np.array([math.cos(i_ang), math.sin(i_ang), 0])

        self.N = configuration['num_trajectory']    # number of trajectories for each linear model
        self.T = configuration['T']                 # time horizon of trajectory for model fitting(0<=t<=self.T)

        # weightings in cost function
        self.cost_param = {
            'wx': np.diag([10.0, 0.01, 0.1]),  # (state_dim, state_dim)
            'wu': np.diag([0.001]),  # (action_dim, action_dim)
        }

        # epsilon
        self.kl_step_mult = configuration['init_kl_step_mult']

        # train data structure
        self.training_data = TrainingData()
        self.prev_training_data = TrainingData()
        self.sampler = Sampler(self.env, self.N, self.T, self.state_dim, self.action_dim)

        # create local dynamic models p(x_t+1|xt,ut)
        self.dynamics_data = DynamicsData()
        self.prev_dynamics_data = DynamicsData()
        self.local_dynamics = LocalDynamics(self.T, self.state_dim, self.action_dim, self.prior)

        # create local Gaussian controller p(ut|xt)
        self.control_data = ControlData()
        self.prev_control_data = ControlData()
        self.local_controller = LocalControl(self.T, self.state_dim, self.action_dim)

        # save the results
        self.save_costs = []

    ## update LQR
    def update(self, MAX_ITER):

        print("Now, regular iteration starts ...")

        for iter in range(int(MAX_ITER)):

            print("\niter = ", iter)

            # step 1: generate training data (by running previous LQR) -------------------
            #  1. initialize local control law
            #  2. generate training trajectory using prev_LQR

            x0 = self.init_state

            if iter == 0:
                self.control_data = self.local_controller.init()
                self.training_data = self.sampler.generate(x0, self.control_data, self.cost_param, self.goal_state)
            else:
                self.training_data = self.sampler.generate(x0, self.prev_control_data, self.cost_param, self.goal_state)

            # evaluate cost
            iter_cost = self.training_data.cost
            self.save_costs.append(iter_cost)
            print("     iter_cost  = ", iter_cost)

            # step 2: fit dynamics ------------------------------------------------------
            #  1. update GMM prior
            #  2. fit local linear time-varying dynamic model

            self.dynamics_data = self.local_dynamics.update(self.training_data)

            # step 3: update trajectory (update LQR) -----------------------------------
            #  1. design local LQR using local dynamic model

            if iter > 0:
                eta = self.prev_control_data.eta
                self.control_data = self.local_controller.update(self.prev_control_data,
                                                                self.dynamics_data, self.cost_param,
                                                                self.goal_state,
                                                                eta, self.kl_step_mult)

            # step 4: adjust kl_step -----------------------------------------------------

            if iter > 0:
                self._epsilon_adjust()

            # step 5: prepare next iteration ---------------------------------------------

            self._update_iteration_variables()


        np.savetxt('./save_weights/pendulum_iter_cost.txt', self.save_costs)


    ## adjust KL step (epsilon)
    def _epsilon_adjust(self):

        _last_cost = self.prev_training_data.cost
        _cur_cost = self.training_data.cost

        _expected_cost = self.estimate_cost(self.control_data, self.dynamics_data)

        # compute predicted and actual improvement
        _expected_impr = _last_cost - _expected_cost
        _actual_impr = _last_cost - _cur_cost

        print("  cost last, expected, current = ", _last_cost, _expected_cost, _cur_cost)

        # adjust epsilon multiplier
        _mult = _expected_impr / (2.0 * max(1e-4, _expected_impr - _actual_impr))
        _mult = max(0.1, min(5.0, _mult))
        new_step = max(
                    min(_mult * self.kl_step_mult, configuration['max_kl_step_mult']),
                        configuration['min_kl_step_mult']
                    )

        self.kl_step_mult = new_step

        print(" epsilon_mult = ", new_step)


    ## previous <-- current
    def _update_iteration_variables(self):

        self.prev_training_data = copy.deepcopy(self.training_data)
        self.prev_dynamics_data = copy.deepcopy(self.dynamics_data)
        self.prev_control_data = copy.deepcopy(self.control_data)

        self.training_data = TrainingData()
        self.dynamics_data = DynamicsData()
        self.control_data = ControlData()


    ## estimate cost with local dynamics and local controller
    def estimate_cost(self, control_data, dynamics_data):

        T, state_dim, action_dim = self.T, self.state_dim, self.action_dim

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)

        # original cost
        Ctt = np.zeros((state_dim + action_dim, state_dim + action_dim))
        Ctt[slice_x, slice_x] = self.cost_param['wx'] * 2.0
        Ctt[slice_u, slice_u] = self.cost_param['wu'] * 2.0
        ct = np.zeros((state_dim + action_dim))
        ct[slice_x] = -2.0 * self.cost_param['wx'].dot(self.goal_state)
        cc = self.goal_state.T.dot(self.cost_param['wx']).dot(self.goal_state)

        # pull out dynamics
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)
        dyn_cov = dynamics_data.dyn_cov  # (T+1, state_dim, state_dim)

        # pull out local controller
        Kt = control_data.Kt
        kt = control_data.kt
        St = control_data.St

        # initialization
        predicted_cost = np.zeros(T+1)
        xu_mu = np.zeros((T+1, state_dim + action_dim))
        xu_cov = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        xu_mu[0, slice_x] = dynamics_data.x0mu
        xu_cov[0, slice_x, slice_x] = dynamics_data.x0cov

        for t in range(T+1):

            xu_mu[t,:] = np.hstack([
                xu_mu[t, slice_x],
                Kt[t,:,:].dot(xu_mu[t, slice_x]) + kt[t, :]
            ])

            xu_cov[t,:,:] = np.vstack([
                np.hstack([
                    xu_cov[t, slice_x, slice_x], xu_cov[t, slice_x, slice_x].dot(Kt[t,:,:].T)
                ]),
                np.hstack([
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]),
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]).dot(Kt[t,:,:].T) + St[t,:,:]
                ])
            ])

            if t < T:
                xu_mu[t+1, slice_x] = fxu[t, :, :].dot(xu_mu[t, :]) + fc[t, :]
                xu_cov[t+1, slice_x, slice_x] = fxu[t,:,:].dot(xu_cov[t,:,:]).dot(fxu[t,:,:].T) + dyn_cov[t,:,:]

        for t in range(T+1):
            x = xu_mu[t, slice_x]
            u = xu_mu[t, slice_u]
            predicted_cost[t] = (x - self.goal_state).T.dot(self.cost_param['wx']).dot(x - self.goal_state) + \
                           u.T.dot(self.cost_param['wu']).dot(u) * np.sum(xu_cov[t, :, :]*Ctt)

        return predicted_cost.sum()

