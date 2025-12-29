from protocol.simple_majority import SimpleMajorityProtocol, State as SimpleMajorityState
from protocol.atomic_commit import AtomicCommitProtocol, State as AtomicCommitState
from protocol.primary_backup import PrimaryBackupProtocol, State as PrimaryBackupState
from groundtruth.simple_majority_human import SimpleMajorityHuman
from groundtruth.atomic_commit_human import AtomicCommitHuman
from groundtruth.primary_backup_human import PrimaryBackupHuman
from simulator.generator import ReadableInputGenerator

import logging

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

def initialize_input_generator(players, rounds, protocol_related: dict, invariance: int) -> ReadableInputGenerator:
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
