# ------------------------------------------------------------------------------
#   set hyperparameters
# ------------------------------------------------------------------------------

configuration = {

    # experiment configuration
    'T': 150,
    'num_trajectory': 20, # 20

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

    'base_kl_step': 0.01,  # epsilon
    'init_kl_step_mult': 1.0,
    'min_kl_step_mult': 1e-1,
    'max_kl_step_mult': 1e2,

    # render
    'render_ok': False,

}
