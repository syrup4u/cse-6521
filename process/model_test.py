import common
import config
from model.environment import Environment
from lib.utils import get_round_info

from protocol import AtomicCommitState

import logging
import os
import torch

logger = logging.getLogger(__name__)

def get_policy(target_dir: str, all_msgs: list, round_info: dict):
    cfg: config.Config = config.load_config(os.path.join(target_dir, config.CONFIG_PATH))
    protocol_related = common.PROTOCOL_TABLE.get(cfg.protocol)
    players = len(all_msgs)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = cfg.train.device
    )
    model = common.initialize_model(cfg, players, round_info['rounds'], os.path.join(target_dir, config.MODEL_PATH))

    def _get_actions(msgs_list: list, round_info: int) -> list:
        state_tensor = env.get_state_all(msgs_list, round_info)
        actions = model.get_greedy_action(state_tensor)
        next_states = env.step_all(actions)
        return next_states
    
    model.eval()
    with torch.inference_mode():
        logger.info(f"--- Round {round_info['current_round']} ---")
        for i, msgs in enumerate(all_msgs):
            logger.info(f"Messages {i}: {msgs}")
        next_states = _get_actions(all_msgs, get_round_info(round_info['current_round'], round_info['rounds'], cfg.model.encode_round_number))
        logger.info(f"Next states: {[s.name for s in next_states]}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    target_dir = "results/ac_p3_r3"
    round_info = {
        'current_round': 3,
        'rounds': 5
    }
    all_msgs = [
        ["Lost", "Lost", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero"],
        ["Lost", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero"],
        ["Lost", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero"],
        ["Lost", "Lost", "Lost", "DoNothing_Zero", "DoNothing_Zero"],
        ["Lost", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero", "DoNothing_Zero"]
    ]
    all_msgs = list(map(lambda x: [AtomicCommitState[s] for s in x], all_msgs))
    get_policy(target_dir, all_msgs, round_info)
