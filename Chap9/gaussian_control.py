## local control law: LQR

import numpy as np
import scipy as sp
from config import configuration

class ControlData(object):

    def __init__(self, Kt=None, kt=None, St=None, chol_St=None, inv_St=None, eta=None):

        self.Kt = Kt                # (T+1, action_dim, state_dim)
        self.kt = kt                # (T+1, action_dim)
        self.St = St                # (T+1, action_dim, action_dim)
        self.chol_St = chol_St      # (T+1, action_dim, action_dim)
        self.inv_St = inv_St        # (T+1, action_dim, action_dim)

        self.eta = eta


class LocalControl(object):

    def __init__(self, T, state_dim, action_dim):

        self.T = T  # t=0, 1, ..., T
        self.state_dim = state_dim
        self.action_dim = action_dim


    ## initialize local Gaussian controller
    def init(self):
        """"
            initialize the local Gaussian controller
        """
        Kt = np.zeros([self.T+1, self.action_dim, self.state_dim])
        kt = np.zeros([self.T+1, self.action_dim])
        St = np.zeros([self.T+1, self.action_dim, self.action_dim])
        chol_St = np.zeros([self.T+1, self.action_dim, self.action_dim])
        inv_St = np.zeros([self.T+1, self.action_dim, self.action_dim])

        T = self.T

        for t in range(T+1):
            St[t, :, :] = 1.0 * np.eye(self.action_dim)
            inv_St[t,:, :] = 1.0 / St[t, :, :]
            chol_St[t, :, :] = sp.linalg.cholesky(St[t, :, :])

        eta = configuration['init_eta']

        return ControlData(Kt, kt, St, chol_St, inv_St, eta)


    ## update local Gaussian LQR
    def update(self, control_data, dynamics_data, cost_param, goal_state, eta,
               kl_step_mult, MAX_iLQR_ITER=20):
        """
        @ description
            update each traj using ilqr.
            if the backward_pass fails, increase
            the kl penalty and recalculate the c_x, c_u, c_uu, c_ux, c_xx
        """

        T = self.T

        max_eta = configuration['max_eta']
        min_eta = configuration['min_eta']

        # set KL bound (epsilon)
        kl_bound = kl_step_mult * configuration['base_kl_step'] * (T+1)

        for itr in range(MAX_iLQR_ITER):

            # LQR backward pass
            backward_pass = self.backward(control_data, dynamics_data, eta, cost_param, goal_state)

            # LQR forward pass
            xu_mu, xu_cov = self.forward(backward_pass, dynamics_data)
            # compute KL divergence between local current and previous control laws

            kl_div = self.trajectory_kl(xu_mu, xu_cov, backward_pass, control_data)

            constraint = kl_div - kl_bound

            if abs(constraint) < 0.1 * kl_bound:
                print("KL converged iteration: ", itr)
                break

            # adjust eta
            if constraint < 0: # eta is too big
                max_eta = backward_pass.eta
                geo_mean = np.sqrt(min_eta*max_eta) # geometric mean
                new_eta = max(geo_mean, 0.1*max_eta)
            else: # eta is too small
                min_eta = backward_pass.eta
                geo_mean = np.sqrt(min_eta*max_eta)
                new_eta = min(10*min_eta, geo_mean)

            eta = new_eta

        return backward_pass

    ## LQR bqckward pass
    def backward(self, control_data, dynamics_data, eta, cost_param, goal_state):
        """
            LQR backward pass
            time-varying linear Gaussian controller
            ut = Kt * xt * kt + act_noise
            x_t+1 = fx * xt + fu ut + fc + noise
            if the backward_pass fails, increase
            the kl penalty and recalculate the cost_param
        """

        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        # pull out dynamics
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)

        # initialization
        Kt = np.zeros((T+1, action_dim, state_dim))
        kt = np.zeros((T+1, action_dim))
        St = np.zeros((T+1, action_dim, action_dim)) # Quut_inv
        chol_St = np.zeros((T+1, action_dim, action_dim))
        Quut = np.zeros((T+1, action_dim, action_dim))

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)
        eta0 = eta
        inc_eta = 1e-4  # in case of non positive-definite of Quu, incremental eta

        Quupd_err = True  # if Quu is non positive-definite matrix
        while Quupd_err:
            Quupd_err = False  # flip to true if Quu is non positive

            # initialization
            Vtt = np.zeros((T+1, state_dim, state_dim))
            vt = np.zeros((T+1, state_dim))
            Qtt = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
            qt = np.zeros((T+1, state_dim + action_dim))

            # compute surrogate costs
            # Ctt: (T+1, state_dim+action_dim, state_dim+action_dim),  ct: (T+1, state_dim+action_dim)
            Ctt, ct = self.augment_cost(control_data, eta, cost_param, goal_state)

            for t in range(T, -1, -1):

                if t == T:
                    Qtt[t] = Ctt[t, :, :]
                    qt[t] = ct[t, :]
                else:
                    Qtt[t] = Ctt[t, :, :] + fxu[t, :, :].T.dot(Vtt[t+1, :, :]).dot(fxu[t, :, :])
                    qt[t] = ct[t, :] + fxu[t, :, :].T.dot(vt[t+1, :] + Vtt[t+1, :, :].dot(fc[t, :]))

                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                Quu = Qtt[t, slice_u, slice_u]
                Qux = Qtt[t, slice_u, slice_x]
                Qu = qt[t, slice_u]

                try:
                    # try to do Cholesky decomposittion for Quu
                    U = sp.linalg.cholesky(Quu)
                    L = U.T
                except:
                    # if decomposition fails, then Quu is not positive definite
                    Quupd_err = True
                    break

                Quut[t, :, :] = Quu
                # compute inverse of Quut
                Quu_inv = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(action_dim), lower=True)
                )
                St[t, :, :] = Quu_inv
                chol_St[t, :, :] = sp.linalg.cholesky(Quu_inv)

                #Kt[t] = -self.St[t].dot(Qtt[t, slice_u, slice_x])
                Kt[t, :, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qux, lower=True)
                )
                #kt[t] = -self.St[t].dot(qt[t, slice_u])
                kt[t, :] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, Qu, lower=True)
                )

                # compute value function
                Vtt[t, :, :] = Qtt[t, slice_x, slice_x] - Qux.T.dot(Quu_inv).dot(Qux)
                Vtt[t, :, :] = 0.5 * (Vtt[t, :, :] + Vtt[t, :, :].T)
                vt[t, :] = qt[t, slice_x] - Qux.T.dot(Quu_inv).dot(Qu)

                #Vtt[t, :, :] = Qtt[t, slice_x, slice_x] + Qtt[t, slice_x, slice_u].dot(Kt[t, :, :])
                #Vtt[t, :, :] = 0.5 * (Vtt[t, :, :] + Vtt[t, :, :].T)
                #vt[t, :] = qt[t, slice_x] + Qtt[t, slice_x, slice_u].dot(kt[t, :])

            # if Quut is not non positive-definite, increment eta
            if Quupd_err:
                eta = eta0 + inc_eta
                inc_eta *= 2.0
                print('Ooops ! Quu is not PD')

                if eta >= 1e16:
                    ValueError('Failed to find PD solution even for very large eta')

        return ControlData(Kt, kt, St, chol_St, Quut, eta)


    ## LQR forward pass
    def forward(self, backward_pass, dynamics_data):
        """
        @ description
            LQR forward pass
            time-varying linear Gaussian controller
            ut = Kt * xt * kt + act_noise
            x_t+1 = fx * xt + fu ut + fc + noise
            output : mu_t+1 = fx * mu_t + f_u * (K_t * mu_t + k_t) + fc
                     X_t+1 = fxu * [X_t X_t *K_t'; K_t*X_t K_t*X_t*K_t'+ Pt] * fxu' + E_t
        """

        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        Kt, kt, St = backward_pass.Kt, backward_pass.kt, backward_pass.St

        # pull out dynamics
        fxu = dynamics_data.fxu            # (T+1, state_dim, (state_dim+action_dim))
        fc = dynamics_data.fc              # (T+1, state_dim)
        dyn_cov = dynamics_data.dyn_cov  # (T+1, state_dim, state_dim)

        # initialization
        xu_mu = np.zeros((T+1, state_dim + action_dim))
        xu_cov = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))

        slice_x = slice(state_dim)
        xu_mu[0, slice_x] = dynamics_data.x0mu
        xu_cov[0, slice_x, slice_x] = dynamics_data.x0cov

        for t in range(T+1):

            xu_mu[t,:] = np.hstack([
                xu_mu[t, slice_x],
                Kt[t,:,:].dot(xu_mu[t, slice_x]) + kt[t, :]
            ])

            xu_cov[t,:,:] = np.vstack([
                np.hstack([
                    xu_cov[t, slice_x, slice_x],
                    xu_cov[t, slice_x, slice_x].dot(Kt[t,:,:].T)
                ]),
                np.hstack([
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]),
                    Kt[t,:,:].dot(xu_cov[t, slice_x, slice_x]).dot(Kt[t,:,:].T) + St[t,:,:]
                ])
            ])

            if t < T:
                xu_mu[t+1, slice_x] = fxu[t, :, :].dot(xu_mu[t, :]) + fc[t, :]
                xu_cov[t+1, slice_x, slice_x] = fxu[t,:,:].dot(xu_cov[t,:,:]).dot(fxu[t,:,:].T) + dyn_cov[t,:,:]

        return xu_mu, xu_cov


    ## compute costs for LQR backward pass for each local model
    def augment_cost(self, policy_data, eta, cost_param, goal_state):
        """
        @ description
            compute augmented cost used in the LQR backward pass.
         IN:
            policy_data = p_bar(ut|xt)
            original cost function = (x-goal_state).T * wx * (x-goal_state) + u.T * wu * u
            dual variable: eta

         OUT:
            modified Dtt (T+1, state_dim + action_dim, state_dim + action_dim)
            modified dt (T+1, state_dim + action_dim)
        """

        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        slice_x = slice(state_dim)
        slice_u = slice(state_dim, state_dim + action_dim)

        # original cost
        Ctt = np.zeros((state_dim + action_dim, state_dim + action_dim))
        Ctt[slice_x, slice_x] = cost_param['wx'] * 2.0
        Ctt[slice_u, slice_u] = cost_param['wu'] * 2.0
        ct = np.zeros(state_dim + action_dim)
        ct[slice_x] = -2.0 * cost_param['wx'].dot(goal_state)

        # initialization
        Hessian = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        Jacobian = np.zeros((T+1, state_dim + action_dim))
        Dtt = np.zeros((T+1, state_dim + action_dim, state_dim + action_dim))
        dt = np.zeros((T+1, state_dim + action_dim))

        for t in range(T+1):
            inv_Sbar = policy_data.inv_St[t,:,:] # (action_dim, state_dim)
            KBar = policy_data.Kt[t, :, :]       # (action_dim, state_dim)
            kbar = policy_data.kt[t, :]          # (action_dim,)

            Hessian[t, :, :] = np.vstack([
                np.hstack([KBar.T.dot(inv_Sbar).dot(KBar), -KBar.T.dot(inv_Sbar)]),
                np.hstack([-inv_Sbar.dot(KBar), inv_Sbar])
            ])  # (state_dim+action_dim, state_dim+action_dim)

            Jacobian[t, :] = np.concatenate([
                KBar.T.dot(inv_Sbar).dot(kbar), -inv_Sbar.dot(kbar)
            ])   # (state_dim+action_dim,)

            Dtt[t,:,:] = Ctt / eta + Hessian[t, :, :]
            dt[t,:] = ct / eta + Jacobian[t, :]

        return Dtt, dt


    ## compute KL divergence between p and p_bar
    def trajectory_kl(self, xu_mu, xu_cov, backward_pass, policy_data):
        """
         KL divergence
        """

        T = self.T
        state_dim = self.state_dim
        action_dim = self.action_dim

        slice_x = slice(state_dim)

        # initialization of KL divergence for each time step
        kl_div_t = np.zeros(T+1)

        # for each time step
        for t in range(T+1):
            inv_Sbar = policy_data.inv_St[t, :, :]
            chol_Sbar = policy_data.chol_St[t, :, :]
            KBar= policy_data.Kt[t, :, :]
            kbar = policy_data.kt[t, :]

            Kt_new = backward_pass.Kt[t, :, :]
            kt_new = backward_pass.kt[t, :]
            St_new = backward_pass.St[t, :, :]
            chol_St_new = backward_pass.chol_St[t, :, :]

            K_diff = KBar - Kt_new
            k_diff = kbar - kt_new

            state_mu = xu_mu[t, slice_x]
            state_cov = xu_cov[t, slice_x, slice_x]

            logdet_Sbar = 2 * sum(np.log(np.diag(chol_Sbar)))
            logdet_St_new = 2 * sum(np.log(np.diag(chol_St_new)))

            kl_div_t[t] = max(
                0,
                0.5 * (
                    np.sum(np.diag(inv_Sbar.dot(St_new))) + \
                    logdet_Sbar - logdet_St_new - action_dim + \
                    k_diff.T.dot(inv_Sbar).dot(k_diff) + \
                    2 * k_diff.T.dot(inv_Sbar).dot(K_diff).dot(state_mu) + \
                    np.sum(np.diag(K_diff.T.dot(inv_Sbar).dot(K_diff).dot(state_cov))) + \
                    state_mu.T.dot(K_diff.T).dot(inv_Sbar).dot(K_diff).dot(state_mu)
                )
            )

        kl_div = np.sum(kl_div_t)

        return kl_div
