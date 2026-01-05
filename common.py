from protocol import *
from groundtruth import *
from simulator.generator import ReadableInputGenerator
import config

import logging
import importlib
import os

logger = logging.getLogger(__name__)

PROTOCOL_TABLE = {
    "simple_majority": {
        "protocol_class": SimpleMajorityProtocol,
        "groundtruth_class": SimpleMajorityHuman,
        "last_round_work": False,
        "state": SimpleMajorityState,
        "state_offset": 2, # offset for action space alignment (positive: 0+offset, negative: length-offset)
        # for generalization
        "pooling_type": "majority",
        "final_offset": 2,
        "special_state": None,
    },
    "atomic_commit": {
        "protocol_class": AtomicCommitProtocol,
        "groundtruth_class": AtomicCommitHuman,
        "last_round_work": True,
        "state": AtomicCommitState,
        "state_offset": -3,
        # for human generalization
        "pooling_type": "any",
        "final_offset": 0,
        "special_state": AtomicCommitState.Abort.value,
    },
    "primary_backup": {
        "protocol_class": PrimaryBackupProtocol,
        "groundtruth_class": PrimaryBackupHuman,
        "last_round_work": True,
        "state": PrimaryBackupState,
        "state_offset": -3,
        # for human generalization
        "pooling_type": "any",
        "final_offset": 0,
        "special_state": PrimaryBackupState.One.value,
    }
}

def initialize_input_generator(protocol_related: dict, players: int, rounds: int, invariance: int) -> ReadableInputGenerator:
    logger.info("Generating all possible input patterns...")
    rig = ReadableInputGenerator(
        num_nodes=players,
        rounds=rounds, 
        legal_initial_state=protocol_related["state"].get_initial_states(), 
        last_round_work=protocol_related["last_round_work"]
    )
    if invariance == 0:
        rig.generate_all_inputs()
    else:
        rig.generate_filtered_inputs(invariance)
    logger.info("Input patterns generated.")
    return rig

def initialize_model(cfg: config.Config, players: int, rounds: int, model_path: str="") :
    algo = importlib.import_module(f"model.{cfg.algorithm.name}")
    state = PROTOCOL_TABLE[cfg.protocol]["state"]
    state_offset = PROTOCOL_TABLE[cfg.protocol]["state_offset"]
    if cfg.model.name == "mlp":
        model = algo.build_mlp_model(
            cfg,
            input_size = players + 1,
            output_size = len(list(state)) - abs(state_offset),
            device = cfg.train.device
        )
    elif cfg.model.name == "mlp_op":
        if cfg.model.encode_round_number:
            one_hot_length = len(state) + rounds
        else: # Only encode states + is_last_round
            one_hot_length = len(state) + 2
        model = algo.build_mlp_op_model(
            cfg,
            one_hot_length = one_hot_length,
            output_size = len(state) - abs(state_offset),
            device = cfg.train.device
        )
    elif cfg.model.name == "set_transformer":
        model = algo.build_set_transformer_model(
            cfg,
            dim_output = len(state) - abs(state_offset),
            num_states = len(state),
            num_rounds = rounds,
            device = cfg.train.device
        )
    if model_path and os.path.exists(model_path):
        model.load_model(model_path)
        logger.info(f"Loaded trained model from {model_path}.")
    return model
