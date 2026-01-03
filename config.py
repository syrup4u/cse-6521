from dataclasses import dataclass, field
from omegaconf import OmegaConf

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

CONFIG_PATH = "config.yaml"
TRAIN_LOG_PATH = "training.log"
EVAL_LOG_PATH = "evaluation.log"
MODEL_PATH = "model.pth"

"""
===========================
For models
===========================
If model is loaded from a checkpoint, you should ensure the configuration matches the saved model.
"""

@dataclass
class MLPConfig:
    hidden_sizes: list = field(default_factory=lambda: [128, 64, 32])

@dataclass
class SetTransformerConfig:
    dim_input: int = 3
    num_inds: int = 4
    dim_hidden: int = 64
    num_heads: int = 2
    num_outputs: int = 1

@dataclass
class ModelConfig:
    name: str = "set_transformer"
    encode_round_number: bool = True
    mlp: MLPConfig = field(default_factory=MLPConfig)
    st: SetTransformerConfig = field(default_factory=SetTransformerConfig)

"""
===========================
For training
===========================
"""

@dataclass
class A2CConfig:
    ppo_epochs: int = 10
    entropy_gamma: float = 0.1
    clip_epsilon: float = 0.1

@dataclass
class DQNConfig:
    loss: str = "SmoothL1Loss"
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.2
    eps_decay: int = 10000

@dataclass
class AlgorithmConfig:
    name: str = "dqn"
    a2c: A2CConfig = field(default_factory=A2CConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    learning_rate: float = 1e-3

@dataclass
class TrainingConfig:
    epochs: int = 100
    episode_repetition: int = 20
    sample_size: int = 100
    sample_ratio: float = 0.4
    invariance_level: int = 3
    device: str = "cpu"

@dataclass
class Config:
    protocol: str = "primary_backup"
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

"""
===========================
Create Config
===========================
"""

def create_default_config(fp):
    config = OmegaConf.structured(Config)
    OmegaConf.save(config, fp)

def load_config(fp):
    return OmegaConf.load(fp)

if __name__ == "__main__":
    create_default_config("default_config.yaml")
