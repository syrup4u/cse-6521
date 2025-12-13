from simulator.state_machine import StateMachine, StateMachineManager
from simulator.generator import get_sender_idx_from_input

from typing import Protocol
import logging

logger = logging.getLogger(__name__)

class SupportProtocol(Protocol):
    @classmethod
    def get_reward(cls, global_state: list[StateMachine]) -> int:
        ...

def verify(protocol: SupportProtocol, global_state: list[StateMachine]) -> bool:
    reward = protocol.get_reward(global_state)
    if reward >= 0:
        return True
    return False

def evaluate_model(args, smm: StateMachineManager, protocol_related: dict, get_actions: callable, target_inputs: list, print_failed: bool = False) -> list:
    # Run evaluation for each input pattern
    failed_cases = []
    accumulate_reward = 0
    for input_pattern in target_inputs:
        logger.debug(f"Evaluating input pattern: {input_pattern}")

        # Init state
        smm.initialize(args.protocol, input_pattern.initial_states)

        # Perform transitions for each round
        for round_idx in range(args.rounds):
            crashed_this_round = input_pattern.crash_pattern[round_idx]
            senders_this_round = get_sender_idx_from_input(input_pattern, round_idx, protocol_related["last_round_work"]) # mailbox
            global_state_this_round = smm.get_global_state()

            # Transition each state machine
            msgs_list = []
            active_idx = []
            for node_idx, sm in enumerate(smm.state_machines):
                if node_idx in crashed_this_round:
                    sm.set_crashed()
                if senders_this_round[node_idx]:
                    msgs = smm.apply_mask(global_state_this_round, senders_this_round[node_idx])
                    logger.debug(f"Node {node_idx} received messages: {msgs}")
                    msgs_list.append(msgs)
                    active_idx.append(node_idx)
                else:
                    sm.transition(protocol_related["state"].get_lost_state(), None)

            actions = get_actions(msgs_list, round_idx)
            for i, node_idx in enumerate(active_idx):
                smm.state_machines[node_idx].transition(actions[i], msgs_list[i])

        logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")

        # Perform verification
        if not verify(protocol_related["protocol_class"], smm.state_machines):
            failed_cases.append(input_pattern)
            logger.debug(f"Verification failed for input pattern: {input_pattern}")
            logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")
            if print_failed:
                for node_idx, sm in enumerate(smm.state_machines):
                    logger.debug(f"Node {node_idx} history states: {sm.history_state}")
                    logger.debug(f"Node {node_idx} history messages: {sm.history_message}")
        # optional: accumulate reward
        accumulate_reward += protocol_related["protocol_class"].get_reward(smm.state_machines)

    # Log evaluation results
    logger.info(f"Evaluation for protocol <{args.protocol}> completed.")
    logger.info(f"Setting: players={args.players}, rounds={args.rounds}")
    logger.info(f"Result: {len(failed_cases)} failed cases out of {len(target_inputs)} total cases.")
    logger.info(f"Accumulated reward: {accumulate_reward}")

    return failed_cases