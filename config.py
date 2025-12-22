SUPPORT_PROTOCOLS = (
    "simple_majority",
    "atomic_commit",
    "primary_backup"
)

SUPPORT_MODELS = (
    "mlp",
    "mlp_op",
    "set_transformer"
)

SUPPORT_ALGORITHMS = (
    "a2c",
    "dqn"
)

"""
===========================
For models
===========================
If model is loaded from a checkpoint, you should ensure the configuration matches the saved model.
"""

ENCODE_ROUND_NUMBER = True # Default: True

MLP_CONFIG = {
    "hidden_sizes": [128, 64, 32]
}

SET_TRANSFORMER_CONFIG = {
    "dim_input": 3, # embedding dimension for each state
    "num_inds": 4,
    "dim_hidden": 64,
    "num_heads": 2,
    "num_outputs": 1
}

"""
===========================
For training
===========================
"""

A2C_CONFIG = {
    "learning_rate": 1e-3,
    "ppo_epochs": 10,
    "entropy_gamma": 0.1,
    "clip_epsilon": 0.1,
}

DQN_CONFIG = {
    "learning_rate": 1e-3,
    "loss": "SmoothL1Loss",
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update_freq": 300,
    "eps_start": 1.0,
    "eps_end": 0.2,
    "eps_decay": 10000
}

EPISODE_REPETITIONS = 20 # TODO: may be computed based on rounds and players
SAMPLE_SIZE = 100
INPUT_INVARIANCE_LEVEL = 3
SAMPLE_PROBABILITY = 0.4