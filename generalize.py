import config
from simulator.generator import get_sender_idx_from_input
from simulator.state_machine import StateMachineManager
from simulator.subset import SubsetManager
import verifier.verifier as verifier
from model.environment import Environment, L2Transformer
from model.decision_maker import build_human_model, build_set_transformer_model
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
    parser = argparse.ArgumentParser(prog="Generalize Protocols")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--players", "-p", type=int, required=True, help="number of players")
    parent_parser.add_argument("--rounds", "-r", type=int, required=True, help="number of rounds")
    parent_parser.add_argument("--protocol", "-P", type=str, required=True, help="protocol type", choices=config.SUPPORT_PROTOCOLS)
    parent_parser.add_argument("--log_level", type=str, default="info", help="logging level", choices=["debug", "info", "warning", "error", "critical"])
    parent_parser.add_argument("--algorithm", type=str, default="a2c", help="algorithm architecture of l1 model", choices=config.SUPPORT_ALGORITHMS)
    parent_parser.add_argument("--model", type=str, default="mlp", help="underlying neural network of l1 model", choices=config.SUPPORT_MODELS)
    parent_parser.add_argument("--load_l1", type=str, default="", help="load path to the layer-1 model")
    parent_parser.add_argument("--load_l2", type=str, default="", help="load path to the layer-2 model")
    parent_parser.add_argument("--device", type=str, default="cpu", help="device to use for training/evaluation", choices=["cpu", "cuda", "mps"])
    parent_parser.add_argument("--gt_l1", action='store_true', help="use ground truth as layer 1")

    parser_train = subparsers.add_parser("train", help="train the model to generalize", parents=[parent_parser])
    parser_train.add_argument("--model_save", type=str, default="", help="save path of the trained model file")
    parser_train.add_argument("--epochs", type=int, default=100, help="number of training epochs as a limit")
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate the trained model on a protocol", parents=[parent_parser])
    parser_evaluate.add_argument("--invariance", type=int, default=0, help="input invariance level for filtering input patterns (0-3)")
    parser_evaluate.add_argument("--human_l2", action='store_true', help="use human-implemented pooling as layer 2")
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    return args

def initialize_l1_model(args, protocol_related: dict):
    assert len(args.load_l1) > 0, "Layer-1 model path must be provided if not using ground truth."
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
    model.load_model(args.load_l1)
    logger.info(f"Loaded l1 trained model from {args.load_l1}.")
    return model

def initialize_l2_model(args, protocol_related: dict):
    human_l2 = getattr(args, "human_l2", None)
    if human_l2:
        model = build_human_model(
            num_states=len(protocol_related["state"].get_final_states()),
            pooling_type=protocol_related["pooling_type"],
            state_offset=protocol_related["final_offset"],
            specified_state=protocol_related["special_state"],
            device=args.device
        )
    else:
        model = build_set_transformer_model(dim_output=len(protocol_related["state"].get_final_states()), device=args.device)
        model.load_model(args.load_l2)
    return model

def train(args):
    smm = StateMachineManager(num_state_machines=args.players)
    subset_manager = SubsetManager(num_state_machines=args.players, subset_size=3)

    logger.info("===== Training Mode =====")
    logger.info(f"Training protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")

    logger.info("Initializing the environment and model...")
    protocol_related = common.PROTOCOL_TABLE.get(args.protocol)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = args.device
    )
    rig = common.initialize_input_generator(args.players, args.rounds, protocol_related, config.INPUT_INVARIANCE_LEVEL)

def evaluate(args):
    smm = StateMachineManager(num_state_machines=args.players)
    subset_manager = SubsetManager(num_state_machines=args.players, subset_size=3)

    logger.info("===== Evaluation Mode =====")
    logger.info(f"Evaluating protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")

    logger.info("Initializing the environment and model...")
    protocol_related = common.PROTOCOL_TABLE.get(args.protocol)
    rig = common.initialize_input_generator(args.players, args.rounds, protocol_related, args.invariance)

    if args.gt_l1:
        l1_policy = lambda msgs_list, round_idx: protocol_related["groundtruth_class"].get_actions(msgs_list, get_round_info(round_idx, args.rounds, False))
    else:
        model = initialize_l1_model(args, protocol_related)
        model.eval()
        env = Environment(
            state_class = protocol_related["state"],
            offset = protocol_related["state_offset"],
            device = args.device
        )
        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states
        l1_policy = _get_actions
    l2_model = initialize_l2_model(args, protocol_related)
    l2_model.eval()
    l2_transformer = L2Transformer(
        state_class = protocol_related["state"],
        offset = protocol_related["final_offset"],
        device = args.device
    )
    def _make_decision(final_states: list) -> list:
        state_tensor = l2_transformer.get_state_all(final_states)
        actions = l2_model.make_decision(state_tensor)
        next_states = l2_transformer.step_all(actions)
        return next_states

    with torch.inference_mode():
        verifier.evaluate_generalization(args, smm, subset_manager, protocol_related, l1_policy, _make_decision, rig.all_inputs)

    logger.info("Evaluation completed successfully.")


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())
    args.func(args)

if __name__ == "__main__":
    main()