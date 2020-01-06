## fitting the linearized dynamic model with linear regression

import numpy as np


class DynamicsData(object):

    def __init__(self, fxu=None, fc=None, dyn_cov=None, x0mu=None, x0cov=None):

        self.fxu = fxu              # (T+1, state_dim, state_dim + action_dim)
        self.fc = fc                # (T+1, state_dim)
        self.dyn_cov = dyn_cov      # (T+1, state_dim, state_dim)

        self.x0mu = x0mu
        self.x0cov = x0cov


class LocalDynamics(object):

    def __init__(self, T, state_dim, action_dim, prior):

        self.T = T
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.prior = prior


    ## linear model update
    def update(self, training_data):

        X = training_data.X
        U = training_data.U
        N = X.shape[0]

        # update prior
        self.prior.update(training_data)

        # fit dynamics
        fxu, fc, dyn_cov = self.fit(X, U)

        # fit x0mu and x0cov
        x0 = X[:, 0, :]
        x0mu = np.mean(x0, axis=0)                 # (state_dim,)
        x0cov = np.diag(np.maximum(np.var(x0, axis=0), 1e-6))

        mu00, Phi0, priorm, n0 = self.prior.initial_state()
        x0cov += Phi0 + (N*priorm) / (N+priorm) * np.outer(x0mu-mu00, x0mu-mu00) / (N+n0)

        return DynamicsData(fxu, fc, dyn_cov, x0mu, x0cov)


    ## fit dynamics
    def fit(self, X, U, cov_reg=1e-6):

        N = X.shape[0]

        fxu = np.zeros([self.T+1, self.state_dim, self.state_dim + self.action_dim])
        fc = np.zeros([self.T+1, self.state_dim])
        dyn_cov = np.zeros([self.T+1, self.state_dim, self.state_dim])

        slice_xu = slice(self.state_dim + self.action_dim)
        slice_xux = slice(self.state_dim + self.action_dim, self.state_dim + self.action_dim + self.state_dim)

        # Fit dynamics with least squares regression.
        dwts = (1.0 / N) * np.ones(N)

        for t in range(self.T+1):

            # xux = [xt;  ut;  x_t+1]
            xux = np.c_[X[:, t, :], U[:, t, :], X[:, t+1, :]] # (N, state_dim+action_dim+state_dim)
            # compute Normal-inverse-Wishart prior
            mu0, Phi, mm, n0 = self.prior.eval(self.state_dim, self.action_dim, xux)

            # Build weights matrix.
            D = np.diag(dwts)

            # compute empirical mean and covariance (IMPORTANT !!)
            xux_mean = np.mean((xux.T * dwts).T, axis=0)   # <--       # ((state_dim+action_dim+state_dim),)
            #xux_mean = np.mean(xux, axis=0)
            diff = xux - xux_mean
            xux_cov = diff.T.dot(D).dot(diff)    # <-- # (state_dim+action_dim+state_dim, state_dim+action_dim+state_dim)
            #xux_cov = diff.T.dot(diff) / N
            xux_cov = 0.5 * (xux_cov + xux_cov.T)

            # MAP estimate of joint distribution
            map_cov = (Phi + N * xux_cov + (N * mm) / (N + mm) * np.outer(xux_mean-mu0, xux_mean-mu0)) / (N + n0)
            map_cov = 0.5 * (map_cov + map_cov.T)
            map_cov[slice_xu, slice_xu] += cov_reg * np.eye(self.state_dim+self.action_dim) # for matrix inverse

            map_mean = (mm * mu0 + n0 * xux_mean) / (mm + n0)
            #map_mean = xux_mean

            # compute model parameters
            fxut = np.linalg.solve(map_cov[slice_xu, slice_xu], map_cov[slice_xu, slice_xux]).T  # (state_dim, state_dim+action_dim)
            fct = map_mean[slice_xux] - fxut.dot(map_mean[slice_xu]) # (state_dim,)

            proc_cov = map_cov[slice_xux, slice_xux] - fxut.dot(map_cov[slice_xu, slice_xu]).dot(fxut.T)

            fxu[t, :, :] = fxut
            fc[t, :] = fct
            dyn_cov[t, :, :] = 0.5 * (proc_cov + proc_cov.T)

        return fxu, fc, dyn_cov
