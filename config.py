SUPPORT_PROTOCOLS = [
    "simple_majority",
    "atomic_commit"
]

MLP_CONFIG = {
    "hidden_sizes": [128, 64, 32],
    "learning_rate": 1e-3
}

SET_TRANSFORMER_CONFIG = {
    "dim_input": 3, # embedding dimension for each state
    "num_inds": 4,
    "dim_hidden": 64,
    "num_heads": 2,
    "num_outputs": 1,
    "learning_rate": 1e-3
}

DQN_CONFIG = {
    "learning_rate": 1e-3,
    "loss": "SmoothL1Loss",
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update_freq": 100,
    "eps_start": 1.0,
    "eps_end": 0.2,
    "eps_decay": 10000
}

EPISODE_REPETITIONS = 20 # TODO: may be computed based on rounds and players
SAMPLE_SIZE = 100
INPUT_INVARIANCE_LEVEL = 3
SAMPLE_PROBABILITY = 0.3