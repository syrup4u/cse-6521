import config
from simulator.generator import get_sender_idx_from_input
from simulator.state_machine import StateMachineManager
import verifier.verifier as verifier
from model.environment import Environment
from lib.utils import sample_from_two_lists, get_round_info
import common

import argparse
import logging
import importlib
import torch

# Set up logging configuration
logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("main")

def parse_args():
    parser = argparse.ArgumentParser(prog="Learn Protocols")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--players", "-p", type=int, required=True, help="number of players")
    parent_parser.add_argument("--rounds", "-r", type=int, required=True, help="number of rounds")
    parent_parser.add_argument("--protocol", "-P", type=str, required=True, help="protocol type", choices=config.SUPPORT_PROTOCOLS)
    parent_parser.add_argument("--log_level", type=str, default="info", help="logging level", choices=["debug", "info", "warning", "error", "critical"])
    parent_parser.add_argument("--model", type=str, default="mlp", help="underlying neural network", choices=config.SUPPORT_MODELS)
    parent_parser.add_argument("--model_load", type=str, default="", help="load path to the trained model file")
    parent_parser.add_argument("--algorithm", type=str, default="a2c", help="training algorithm", choices=config.SUPPORT_ALGORITHMS)
    parent_parser.add_argument("--device", type=str, default="cpu", help="device to use for training/evaluation", choices=["cpu", "cuda", "mps"])

    parser_train = subparsers.add_parser("train", help="train the model to learn a protocol", parents=[parent_parser])
    parser_train.add_argument("--model_save", type=str, default="", help="save path of the trained model file")
    parser_train.add_argument("--epochs", type=int, default=100, help="number of training epochs as a limit") # TODO: move to config
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate the trained model on a protocol", parents=[parent_parser])
    parser_evaluate.add_argument("--groundtruth", "-gt", action='store_true', help="use ground truth (human designed) for evaluation")
    parser_evaluate.add_argument("--invariance", type=int, default=0, help="input invariance level for filtering input patterns (0-3)")
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    return args

def initialize_model(args, protocol_related: dict):
    algo = importlib.import_module(f"model.{args.algorithm}")
    if args.model == "mlp":
        model = algo.build_mlp_model(
            input_size = args.players + 1,
            output_size = len(list(protocol_related["state"])) - abs(protocol_related["state_offset"]),
            device = args.device
        )
    elif args.model == "mlp_op":
        if config.ENCODE_ROUND_NUMBER:
            one_hot_length = len(protocol_related["state"]) + args.rounds
        else: # Only encode states + is_last_round
            one_hot_length = len(protocol_related["state"]) + 2
        model = algo.build_mlp_op_model(
            one_hot_length = one_hot_length,
            output_size = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
            device = args.device
        )
    elif args.model == "set_transformer":
        model = algo.build_set_transformer_model(
            dim_output = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
            num_states = len(protocol_related["state"]),
            num_rounds = args.rounds,
            device = args.device
        )
    if args.model_load:
        model.load_model(args.model_load)
        logger.info(f"Loaded trained model from {args.model_load}.")
    return model

# TODO: multiprocessing and sampling for evaluation
def evaluate(args, smm: StateMachineManager):
    logger.info("===== Evaluation Mode =====")

    logger.info(f"Evaluating protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")
    protocol_related = common.PROTOCOL_TABLE.get(args.protocol)

    rig = common.initialize_input_generator(args.players, args.rounds, protocol_related, args.invariance)

    if args.groundtruth:
        logger.info("Using ground truth for evaluation.")
        protocol_related["groundtruth_class"].check_rounds(args.rounds)
        logger.info("Ground truth protocol check passed.")

        action_func = protocol_related["groundtruth_class"].get_actions
        if protocol_related["last_round_work"]:
            action_func = lambda msgs_list, round_idx: protocol_related["groundtruth_class"].get_actions(msgs_list, get_round_info(round_idx, args.rounds, False))

        verifier.evaluate_model(args, smm, protocol_related, action_func, rig.all_inputs)
    else:
        logger.info("Using trained model for evaluation.")
        env = Environment(
            state_class = protocol_related["state"],
            offset = protocol_related["state_offset"],
            device = args.device
        )
        model = initialize_model(args, protocol_related)

        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states

        model.eval()
        with torch.inference_mode():
            verifier.evaluate_model(args, smm, protocol_related, _get_actions, rig.all_inputs)

    logger.info("Evaluation completed successfully.")


def train(args, smm: StateMachineManager):
    logger.info("===== Training Mode =====")
    logger.info(f"Training protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")

    logger.info("Initializing the environment and model...")
    protocol_related = common.PROTOCOL_TABLE.get(args.protocol)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = args.device
    )
    model = initialize_model(args, protocol_related)
    algo = importlib.import_module(f"model.{args.algorithm}")

    rig = common.initialize_input_generator(args.players, args.rounds, protocol_related, config.INPUT_INVARIANCE_LEVEL)

    logger.info("Starting training...")
    # Training parameters
    repetition = config.EPISODE_REPETITIONS
    epochs = args.epochs
    sample_size = config.SAMPLE_SIZE
    sample_probability = config.SAMPLE_PROBABILITY
    failed_inputs = rig.all_inputs
    cur_epoch = 0

    while cur_epoch < epochs:
        cur_epoch += 1
        logger.info(f"Epoch {cur_epoch}/{epochs}, training on {len(failed_inputs)} failed cases from last evaluation.")
        for _ in range(sample_size):
            input_pattern = sample_from_two_lists(rig.all_inputs, failed_inputs, sample_probability)
            logger.debug(f"Training on input pattern: {input_pattern}")
            train_one_case(args, algo, protocol_related, smm, env, model, input_pattern, repetition=repetition)
        
        logger.info("Training completed. Move to evaluation phase.")

        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states
        
        model.eval()
        with torch.inference_mode():
            failed_inputs = verifier.evaluate_model(args, smm, protocol_related, _get_actions, failed_inputs)
            if len(failed_inputs) == 0:
                logger.info("All failed cases passed verification!")
                logger.info("Move to final evaluation on all inputs.")
                failed_inputs = verifier.evaluate_model(args, smm, protocol_related, _get_actions, rig.all_inputs)
                if len(failed_inputs) == 0:
                    logger.info("All cases passed verification! Training successful.")
                    break
                logger.info(f"{len(failed_inputs)} failed cases remain after final evaluation.")
        model.train()

    if args.model_save:
        logger.info(f"Saving trained model to {args.model_save}...")
        model.save_model(args.model_save)
        logger.info("Model saved successfully.")


def train_one_case(args, algo, protocol_related, smm: StateMachineManager, env: Environment, model, input_pattern, repetition: int = 5):
    reward_list = []
    trajectories = dict()

    # Repeat rollout for the same input pattern 
    # (to reduce variance / increase efficiency of exploration)
    for _ in range(repetition):
        # Init state
        smm.initialize(args.protocol, input_pattern.initial_states)

        # Perform transitions for each round (rollout)
        for round_idx in range(args.rounds):
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
            state_tensor = env.get_state_all(msgs_list, get_round_info(round_idx, args.rounds, config.ENCODE_ROUND_NUMBER))
            actions, extra = model.get_action(state_tensor)
            next_states = env.step_all(actions)
            if round_idx < args.rounds - 1:
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
    algo.train_model(model, trajectories=trajectories, rewards=reward_list, others={"episodes": repetition})


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    # Build state machine manager for simulation
    smm = StateMachineManager(num_state_machines=args.players)

    args.func(args, smm)

if __name__ == "__main__":
    main()
