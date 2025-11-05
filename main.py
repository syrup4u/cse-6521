import config
from groundtruth.simple_majority_human import SimpleMajorityHuman
from simulator.generator import ReadableInputGenerator, get_sender_idx_from_input
from simulator.state_machine import StateMachineManager
from protocol.simple_majority import SimpleMajorityProtocol, State as SimpleMajorityState
import verifier.verifier as verifier

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
        "state": SimpleMajorityState
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", "-p", type=int, required=True, help="number of players")
    parser.add_argument("--rounds", "-r", type=int, required=True, help="number of rounds")
    parser.add_argument("--protocol", "-P", type=str, required=True, help="protocol type", choices=config.SUPPORT_PROTOCOLS)
    parser.add_argument("--groundtruth", "-gt", action='store_true', help="use ground truth (human designed) for evaluation")
    parser.add_argument("--evaluate", action='store_true', help="evaluate the protocol")
    args = parser.parse_args()
    return args

# TODO: multiprocessing and sampling for evaluation
def evaluate(args, smm: StateMachineManager):
    assert args.groundtruth, "Ground truth evaluation is currently the only supported mode."

    logger.info(f"Evaluating protocol: {args.protocol} with {args.players} players and {args.rounds} rounds.")
    protocol_related = PROTOCOL_TABLE.get(args.protocol)

    if args.groundtruth:
        logger.info("Using ground truth for evaluation.")
        if not protocol_related["groundtruth_class"].check_rounds(args.rounds):
            logger.error(f"{args.protocol} protocol only supports one round.")
            return
        logger.info("Ground truth protocol check passed.")

        logger.info("Generating all possible input patterns...")
        rig = ReadableInputGenerator(num_nodes=args.players, 
                                     rounds=args.rounds, 
                                     legal_initial_state=protocol_related["state"].get_initial_states(), 
                                     last_round_work=protocol_related["last_round_work"])
        rig.generate_all_inputs()
        logger.info("Input patterns generated.")

        # Run evaluation for each input pattern
        failed_cases = 0
        for input_pattern in rig.all_inputs:
            logger.debug(f"Evaluating input pattern: {input_pattern}")
            # Init state
            smm.initialize(args.protocol, input_pattern.initial_states)
            # Perform transitions for each round
            for round_idx in range(args.rounds):
                crashed_this_round = input_pattern.crash_pattern[round_idx]
                senders_this_round = get_sender_idx_from_input(input_pattern, round_idx, protocol_related["last_round_work"])
                global_state_this_round = smm.get_global_state()
                # Transition each state machine
                for node_idx, sm in enumerate(smm.state_machines):
                    if node_idx in crashed_this_round:
                        sm.set_crashed()
                    if senders_this_round[node_idx]:
                        msgs = smm.apply_mask(global_state_this_round, senders_this_round[node_idx])
                        action = protocol_related["groundtruth_class"].get_action(sm, msgs)
                        sm.transition(action, msgs)
                    else:
                        sm.transition(protocol_related["state"].get_lost_state(), None)
            logger.debug(f"Final global state: {[sm.get_final_state() for sm in smm.state_machines]}")
            # Perform verification
            if not verifier.verify(args.protocol, smm.state_machines):
                failed_cases += 1
                logger.warning(f"Verification failed for input pattern: {input_pattern}")

        # Log evaluation results
        logger.info(f"Evaluation for protocol <{args.protocol}> (ground truth) completed.")
        logger.info(f"Setting: players={args.players}, rounds={args.rounds}")
        logger.info(f"Result: {failed_cases} failed cases out of {len(rig.all_inputs)} total cases.")

    # TODO: Evaluation based on the model

    logger.info("Evaluation completed successfully.")

def main():
    args = parse_args()

    # Build state machine manager for simulation
    smm = StateMachineManager(num_state_machines=args.players)

    if args.evaluate:
        evaluate(args, smm)

if __name__ == "__main__":
    main()
