+130 ~ -130 까지 작동      ------------- keep4
ITER=60
i_ang = np.array([-50.0]) * np.pi / 180.0
self.cost_param = {
            'wx': np.diag([1.0, 0.01, 0.1]),  # (state_dim, state_dim)
            'wu': np.diag([0.001]),  # (action_dim, action_dim)
        }
kt[t, :] = 0. * np.ones(self.action_dim)
St[t, :, :] = 1.0 * np.eye(self.action_dim)

 # experiment configuration
    'T': 150,
    'num_trajectory': 20, # 10

    # GMM configuration
    'gmm_max_samples': 20, # 20
    'gmm_max_clusters': 20,
    'gmm_min_samples_per_cluster': 40, # 40
    'gmm_prior_strength': 1.0,

    # eta, epsilon adjustment
    'init_eta': 1.0,
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'eta_multiplier': 1e-4,

    'base_kl_step': 0.01,  # 25
    'init_kl_step_mult': 1.0,
    'min_kl_step_mult': 1e-1,
    'max_kl_step_mult': 1e2,

    # network training
    'learning_rate': 0.001,
    'training_epochs': 10,


+150 ~ -150 까지 작동      ------------- keep5
* SME PARAMETERS except
self.cost_param = {
            'wx': 2.0*np.diag([1.0, 0.01, 0.1]),  # (state_dim, state_dim)
            'wu': np.diag([0.001]),  # (action_dim, action_dim)
        }

LQR works for +180 ~ -180      ------------- keep6
* SME PARAMETERS except
i_ang = np.array([-45.0 ]) * np.pi / 180.0
 self.cost_param = {
            'wx': 1.0*np.diag([10.0, 0.01, 0.1]),  # (state_dim, state_dim)
            'wu': np.diag([0.001]),  # (action_dim, action_dim)
        }

LQR works for +180 ~ -180      ------------- keep7
* SAME to keep6

LQR works for +180 ~ -180      ------------- keep8
* SAME to keep6 BUT m=1, n0=1, map_mean = 원래 식