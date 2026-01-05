import common
import config
from lib.utils import sample_from_two_lists, get_round_info
from model.environment import Environment
from simulator.generator import get_sender_idx_from_input
from simulator.state_machine import StateMachineManager
import verifier

import importlib
import logging
import torch

logger = logging.getLogger(__name__)

def _evaluate_model(
        cfg: config.Config,
        smm: StateMachineManager,
        protocol_related: dict,
        get_actions: callable,
        target_inputs: list
    ) -> list:
    assert len(target_inputs) > 0, "No input patterns to evaluate."

    # Run evaluation for each input pattern
    failed_cases = []
    accumulate_reward = 0
    rounds = len(target_inputs[0].crash_pattern)
    players = len(target_inputs[0].initial_states)
    for input_pattern in target_inputs:
        # logger.debug(f"Evaluating input pattern: {input_pattern}")

        # Init state
        smm.initialize(cfg.protocol, input_pattern.initial_states)

        # Perform transitions for each round
        for round_idx in range(rounds):
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
                    # logger.debug(f"Node {node_idx} received messages: {msgs}")
                    msgs_list.append(msgs)
                    active_idx.append(node_idx)
                else:
                    sm.transition(protocol_related["state"].get_lost_state(), None)

            actions = get_actions(msgs_list, get_round_info(round_idx, rounds, cfg.model.encode_round_number))
            for i, node_idx in enumerate(active_idx):
                smm.state_machines[node_idx].transition(actions[i], msgs_list[i])

        # logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")

        # Perform verification
        if not verifier.verify(protocol_related["protocol_class"], smm.state_machines):
            failed_cases.append(input_pattern)
            logger.debug(f"Verification failed for input pattern: {input_pattern}")
            logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")
            for node_idx, sm in enumerate(smm.state_machines):
                logger.debug(f"Node {node_idx} history states: {sm.history_state}")
                logger.debug(f"Node {node_idx} history messages: {sm.history_message}")
        # optional: accumulate reward
        accumulate_reward += protocol_related["protocol_class"].get_reward(smm.state_machines)

    # Log evaluation results
    logger.info(f"Evaluation for protocol <{cfg.protocol}> completed.")
    logger.info(f"Setting: players={players}, rounds={rounds}")
    logger.info(f"Result: {len(failed_cases)} failed cases out of {len(target_inputs)} total cases.")
    logger.info(f"Accumulated reward: {accumulate_reward}")

    return failed_cases


def _train_one_case(cfg: config.Config, rounds: int, algo, protocol_related: dict, smm: StateMachineManager, env: Environment, model, input_pattern):
    reward_list = []
    trajectories = dict()

    # Repeat rollout for the same input pattern 
    # (to reduce variance / increase efficiency of exploration)
    for _ in range(cfg.train.episode_repetition):
        # Init state
        smm.initialize(cfg.protocol, input_pattern.initial_states)

        # Perform transitions for each round (rollout)
        for round_idx in range(rounds):
            # Known information this round
            crashed_this_round = input_pattern.crash_pattern[round_idx]
            senders_this_round = get_sender_idx_from_input(input_pattern, round_idx, protocol_related["last_round_work"]) # mailbox
            global_state_this_round = smm.get_global_state()

            # Get global state
            msgs_list = []
            active_idx = []
            for node_idx, sm in enumerate(smm.state_machines):
                if node_idx in crashed_this_round:
                    sm.set_crashed()
                if senders_this_round[node_idx]:
                    msgs = smm.apply_mask(global_state_this_round, senders_this_round[node_idx])
                    # logger.debug(f"Node {node_idx} received messages: {msgs}")
                    msgs_list.append(msgs)
                    active_idx.append(node_idx)
                else:
                    sm.transition(protocol_related["state"].get_lost_state(), None)

            # Rollout
            state_tensor = env.get_state_all(msgs_list, get_round_info(round_idx, rounds, cfg.model.encode_round_number))
            actions, extra = model.get_action(state_tensor)
            next_states = env.step_all(actions)
            if round_idx < rounds - 1:
                done = [False] * len(active_idx)
                crashed_next_round = input_pattern.crash_pattern[round_idx + 1]
                for crashed_node in crashed_next_round:
                    done[active_idx.index(crashed_node)] = True
            else:
                done = [True] * len(active_idx)

            # Transition each state machine
            for i, node_idx in enumerate(active_idx):
                smm.state_machines[node_idx].transition(next_states[i], msgs_list[i])

            algo.update_trajectories(trajectories, state_tensor, actions, [extra, done])

        # logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")

        # Compute reward
        reward = protocol_related["protocol_class"].get_reward(smm.state_machines)
        algo.update_trajectories(trajectories)
        reward_list.append(reward)

    # Backpropagation
    algo.train_model(cfg, model, trajectories=trajectories, rewards=reward_list, others={"episodes": cfg.train.episode_repetition})


def evaluate(
        cfg: config.Config,
        rounds: int,
        players: int,
        others: dict={},
    ):
    """
    others:
    - use_groundtruth: bool
    - invariance_level: int
    - model_path: str
    """
    logger.info("===== Evaluation Mode =====")
    logger.info(f"Evaluating protocol: {cfg.protocol} with {players} players and {rounds} rounds.")
    logger.info(f"Algorithm: {cfg.algorithm.name}, Model: {cfg.model.name}")
    logger.info("===========================")

    smm = StateMachineManager(num_state_machines=players)
    protocol_related = common.PROTOCOL_TABLE.get(cfg.protocol)
    rig = common.initialize_input_generator(protocol_related, players, rounds, others.get("invariance_level", 0))

    if others.get("use_groundtruth", False):
        logger.info("Using ground truth for evaluation.")
        protocol_related["groundtruth_class"].check_rounds(rounds)
        logger.info("Ground truth protocol check passed.")

        action_func = protocol_related["groundtruth_class"].get_actions
        if protocol_related["last_round_work"]:
            action_func = lambda msgs_list, round_idx: protocol_related["groundtruth_class"].get_actions(msgs_list, get_round_info(round_idx, rounds, False))

        _evaluate_model(cfg, smm, protocol_related, action_func, rig.all_inputs)
    else:
        logger.info("Using trained model for evaluation.")
        env = Environment(
            state_class = protocol_related["state"],
            offset = protocol_related["state_offset"],
            device = cfg.train.device
        )
        model = common.initialize_model(cfg, players, rounds, others.get("model_path", None))

        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states

        model.eval()
        with torch.inference_mode():
            _evaluate_model(cfg, smm, protocol_related, _get_actions, rig.all_inputs)

    logger.info("Evaluation completed successfully.")


def train(
        cfg: config.Config,
        rounds: int,
        players: int,
        others: dict={},
    ):
    """
    others:
    - model_path: str
    """
    logger.info("===== Training Mode =====")
    logger.info(f"Training protocol: {cfg.protocol} with {players} players and {rounds} rounds.")
    logger.info(f"Algorithm: {cfg.algorithm.name}, Model: {cfg.model.name}")
    logger.info("=========================")

    logger.info("Initializing the environment and model...")
    smm = StateMachineManager(num_state_machines=players)
    protocol_related = common.PROTOCOL_TABLE.get(cfg.protocol)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = cfg.train.device
    )
    model_path = others.get("model_path", None)
    model = common.initialize_model(cfg, players, rounds, model_path)
    algo = importlib.import_module(f"model.{cfg.algorithm.name}")

    rig = common.initialize_input_generator(protocol_related, players, rounds, cfg.train.invariance_level)

    logger.info("Starting training...")
    # Training parameters
    epochs = cfg.train.epochs
    sample_size = cfg.train.sample_size
    sample_probability = cfg.train.sample_ratio
    failed_inputs = rig.all_inputs
    cur_epoch = 0
    success = False

    while cur_epoch < epochs:
        cur_epoch += 1
        logger.info(f"Epoch {cur_epoch}/{epochs}, training on {len(failed_inputs)} failed cases from last evaluation.")
        for _ in range(sample_size):
            input_pattern = sample_from_two_lists(rig.all_inputs, failed_inputs, sample_probability)
            logger.debug(f"Training on input pattern: {input_pattern}")
            _train_one_case(cfg, rounds, algo, protocol_related, smm, env, model, input_pattern)
        
        logger.info("Training completed. Move to evaluation phase.")

        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states
        
        model.eval()
        with torch.inference_mode():
            failed_inputs = _evaluate_model(cfg, smm, protocol_related, _get_actions, failed_inputs)
            if len(failed_inputs) == 0:
                logger.info("All failed cases passed verification!")
                logger.info("Move to final evaluation on all inputs.")
                failed_inputs = _evaluate_model(cfg, smm, protocol_related, _get_actions, rig.all_inputs)
                if len(failed_inputs) == 0:
                    logger.info("All cases passed verification! Training successful.")
                    success = True
                    break
                logger.info(f"{len(failed_inputs)} failed cases remain after final evaluation.")
        model.train()

    if model_path:
        logger.info(f"Saving trained model to {model_path}...")
        model.save_model(model_path)
        logger.info("Model saved successfully.")

    return success, cur_epoch
