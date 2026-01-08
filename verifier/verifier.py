from simulator.state_machine import StateMachine

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

# def evaluate_generalization(
#     args,
#     smm: StateMachineManager,
#     subset_manager: SubsetManager,
#     protocol_related: dict,
#     l1_policy: callable,
#     l2_make_decision: callable,
#     target_inputs: list
# ):
#     failed_cases = []
#     for input_pattern in target_inputs:
#         # logger.debug(f"Evaluating input pattern: {input_pattern}")

#         # Init state
#         smm.initialize(args.protocol, input_pattern.initial_states)
#         subset_manager.init_states(input_pattern.initial_states)

#         # L1: Perform transitions for each round
#         for round_idx in range(args.rounds):
#             crashed_this_round = input_pattern.crash_pattern[round_idx]
#             senders_this_round = get_sender_idx_from_input(input_pattern, round_idx, protocol_related["last_round_work"]) # mailbox
#             for crashed_idx in crashed_this_round:
#                 smm.state_machines[crashed_idx].set_crashed()
#             # Transition each state machine
#             msgs_list = subset_manager.get_msgs_with_mask(senders_this_round)
#             next_states = l1_policy(msgs_list, get_round_info(round_idx, args.rounds, config.ENCODE_ROUND_NUMBER))
#             subset_manager.apply_states(next_states)

#         # L2: Update state machines' final states based on subset states
#         final_subset_states = subset_manager.get_all_final_states()
#         final_states = l2_make_decision(final_subset_states)
#         last_round_crashed = input_pattern.crash_pattern[-1] if protocol_related["last_round_work"] else []
#         for node_idx, sm in enumerate(smm.state_machines):
#             if sm.crashed and node_idx not in last_round_crashed:
#                 sm.transition(protocol_related["state"].get_lost_state(), None)
#             else:
#                 sm.transition(final_states[node_idx], final_subset_states[node_idx])

#         # logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")

#         # Perform verification
#         if not verify(protocol_related["protocol_class"], smm.state_machines):
#             failed_cases.append(input_pattern)
#             logger.debug(f"Verification failed for input pattern: {input_pattern}")
#             logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")
#             for node_idx, sm in enumerate(smm.state_machines):
#                 logger.debug(f"Node {node_idx} history states: {sm.history_state}")
#                 logger.debug(f"Node {node_idx} history messages: {sm.history_message}")

#     # Log evaluation results
#     logger.info(f"Evaluation for protocol <{args.protocol}> completed.")
#     logger.info(f"Setting: players={args.players}, rounds={args.rounds}")
#     logger.info(f"Result: {len(failed_cases)} failed cases out of {len(target_inputs)} total cases.")

#     return failed_cases
