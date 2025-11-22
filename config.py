SUPPORT_PROTOCOLS = [
    "simple_majority",
    "atomic_commit"
]

EPISODE_REPETITIONS = 1000 # TODO: may be computed based on rounds and players

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