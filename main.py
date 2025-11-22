import config
from groundtruth.simple_majority_human import SimpleMajorityHuman
from groundtruth.atomic_commit_human import AtomicCommitHuman
from simulator.generator import ReadableInputGenerator, get_sender_idx_from_input
from simulator.state_machine import StateMachineManager
from protocol.simple_majority import SimpleMajorityProtocol, State as SimpleMajorityState
from protocol.atomic_commit import AtomicCommitProtocol, State as AtomicCommitState
import verifier.verifier as verifier
from model.environment import Environment
from model.a2c import build_mlp_model, build_mlp_op_model, build_set_transformer_model, train_model, train_model_batch

import argparse
import logging

# Set up logging configuration
logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s', level=logging.INFO)

logger = logging.getLogger("main")
PROTOCOL_TABLE = {
    "simple_majority": {
        "protocol_class": SimpleMajorityProtocol,
        "groundtruth_class": SimpleMajorityHuman,
        "last_round_work": False,
        "state": SimpleMajorityState,
        "state_offset": 2 # offset for action space alignment (positive: 0+offset, negative: length-offset)
    },
    "atomic_commit": {
        "protocol_class": AtomicCommitProtocol,
        "groundtruth_class": AtomicCommitHuman,
        "last_round_work": True,
        "state": AtomicCommitState,
        "state_offset": -3
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", "-p", type=int, required=True, help="number of players")
    parser.add_argument("--rounds", "-r", type=int, required=True, help="number of rounds")
    parser.add_argument("--protocol", "-P", type=str, required=True, help="protocol type", choices=config.SUPPORT_PROTOCOLS)
    parser.add_argument("--groundtruth", "-gt", action='store_true', help="use ground truth (human designed) for evaluation")
    parser.add_argument("--evaluate", action='store_true', help="evaluate the protocol")
    parser.add_argument("--log_level", type=str, default="info", help="logging level", choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--model", type=str, default="mlp", help="path to the trained model", choices=["mlp", "mlp_op", "set_transformer"])
    parser.add_argument("--model_path", type=str, default="", help="path to the trained model file")
    parser.add_argument("--algorithm", type=str, default="a2c", help="training algorithm", choices=["a2c"])
    parser.add_argument("--device", type=str, default="cpu", help="device to use for training/evaluation", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()
    return args

# TODO: multiprocessing and sampling for evaluation
def evaluate(args, smm: StateMachineManager):
    logger.info("===== Evaluation Mode =====")

    logger.info(f"Evaluating protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")
    protocol_related = PROTOCOL_TABLE.get(args.protocol)

    if args.groundtruth:
        logger.info("Using ground truth for evaluation.")
        protocol_related["groundtruth_class"].check_rounds(args.rounds)
        logger.info("Ground truth protocol check passed.")

        action_func = protocol_related["groundtruth_class"].get_actions
        if protocol_related["last_round_work"]:
            action_func = lambda msgs_list, round_idx: protocol_related["groundtruth_class"].get_actions(msgs_list, 1 if round_idx == args.rounds - 1 else 0)

        verifier.evaluate_model(args, smm, protocol_related, action_func)

    logger.info("Evaluation completed successfully.")


def train(args, smm: StateMachineManager):
    logger.info("===== Training Mode =====")
    logger.info(f"Training protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")

    # Initialize
    protocol_related = PROTOCOL_TABLE.get(args.protocol)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = args.device
    )
    if args.algorithm == "a2c":
        if args.model == "mlp":
            model = build_mlp_model(
                input_size = args.players + 1,
                output_size = len(list(protocol_related["state"])) - abs(protocol_related["state_offset"]),
                device = args.device
            )
        elif args.model == "mlp_op":
            model = build_mlp_op_model(
                one_hot_length = len(protocol_related["state"]) + args.rounds,
                output_size = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
                device = args.device
            )
        elif args.model == "set_transformer":
            model = build_set_transformer_model(
                dim_output = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
                num_states = len(protocol_related["state"]),
                num_rounds = args.rounds,
                device = args.device
            )

    logger.info("Generating all possible input patterns for training...")
    rig = ReadableInputGenerator(
        num_nodes=args.players,
        rounds=args.rounds, 
        legal_initial_state=protocol_related["state"].get_initial_states(), 
        last_round_work=protocol_related["last_round_work"]
    )
    rig.generate_all_inputs()
    logger.info("Input patterns generated.")

    def _get_actions(msgs_list: list, round_idx: int) -> list:
        state_tensor = env.get_state_all(msgs_list, round_idx)
        actions = model.get_greedy_action(state_tensor)
        next_states = env.step_all(actions)
        return next_states

    logger.info("Starting training...")
    # TODO: move this part to another function
    repetition = 2
    epochs = 10
    for input_pattern in rig.all_inputs:
        logger.debug(f"Training on input pattern: {input_pattern}")

        traj_list = []
        reward_list = []
        for _ in range(repetition):
            # Init state
            smm.initialize(args.protocol, input_pattern.initial_states)

            # Perform transitions for each round (rollout)
            trajectories = []
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
                        logger.debug(f"Node {node_idx} received messages: {msgs}")
                        msgs_list.append(msgs)
                        active_idx.append(node_idx)
                    else:
                        sm.transition(protocol_related["state"].get_lost_state(), None)

                # Rollout
                state_tensor = env.get_state_all(msgs_list, round_idx)
                actions, log_probs = model.get_action(state_tensor)
                values = model.get_value(state_tensor)
                next_states = env.step_all(actions)

                # Transition each state machine
                for i, node_idx in enumerate(active_idx):
                    smm.state_machines[node_idx].transition(next_states[i], msgs_list[i])
                
                trajectories.append((state_tensor, actions, log_probs, values))

            logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")

            # Compute reward
            reward = protocol_related["protocol_class"].get_reward(smm.state_machines)
            logger.debug(f"Reward obtained: {reward}")
            traj_list.append(trajectories)
            reward_list.append(reward)

        # Backpropagation
        # train_model(model, trajectories, reward)
        train_model_batch(model, traj_list, reward_list)

    logger.info("Training completed. Move to evaluation phase.")
    
    verifier.evaluate_model(args, smm, protocol_related, _get_actions)


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level.upper())

    # Build state machine manager for simulation
    smm = StateMachineManager(num_state_machines=args.players)

    if args.evaluate:
        evaluate(args, smm)
    else:
        train(args, smm)

if __name__ == "__main__":
    main()
