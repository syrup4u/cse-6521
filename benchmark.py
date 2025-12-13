"""
This is a benchmark for evaluating the effectiveness and performance of
the whole framework, including various algorithms / models / methods.
"""

from protocol.simple_majority import SimpleMajorityProtocol, State as SimpleMajorityState
from protocol.atomic_commit import AtomicCommitProtocol, State as AtomicCommitState
from simulator.generator import ReadableInputGenerator, get_sender_idx_from_input
from model.environment import Environment
from simulator.state_machine import StateMachineManager
from lib.sample import sample_from_two_lists
from verifier import verifier
import config

import importlib
import logging
from itertools import product
import pickle

logging.basicConfig(format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s', level=logging.INFO)

logger = logging.getLogger("benchmark")
PROTOCOL_TABLE = {
    "simple_majority": {
        "protocol_class": SimpleMajorityProtocol,
        "last_round_work": False,
        "state": SimpleMajorityState,
        "state_offset": 2 # offset for action space alignment (positive: 0+offset, negative: length-offset)
    },
    "atomic_commit": {
        "protocol_class": AtomicCommitProtocol,
        "last_round_work": True,
        "state": AtomicCommitState,
        "state_offset": -3
    }
}

def initialize_input_generator(players, rounds, protocol_related: dict) -> ReadableInputGenerator:
    rig = ReadableInputGenerator(
        num_nodes=players,
        rounds=rounds, 
        legal_initial_state=protocol_related["state"].get_initial_states(), 
        last_round_work=protocol_related["last_round_work"]
    )
    return rig

def initialize_model(algo, model_name, players, rounds, protocol_related: dict, device='cpu'):
    if model_name == "mlp":
        model = algo.build_mlp_model(
            input_size = players + 1,
            output_size = len(list(protocol_related["state"])) - abs(protocol_related["state_offset"]),
            device = device
        )
    elif model_name == "mlp_op":
        model = algo.build_mlp_op_model(
            one_hot_length = len(protocol_related["state"]) + rounds,
            output_size = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
            device = device
        )
    elif model_name == "set_transformer":
        model = algo.build_set_transformer_model(
            dim_output = len(protocol_related["state"]) - abs(protocol_related["state_offset"]),
            num_states = len(protocol_related["state"]),
            num_rounds = rounds,
            device = device
        )
    return model

def evaluate_model(protocol, rounds, smm: StateMachineManager, protocol_related: dict, get_actions: callable, target_inputs: list) -> list:
    # Run evaluation for each input pattern
    failed_cases = []
    accumulate_reward = 0
    for input_pattern in target_inputs:
        # Init state
        smm.initialize(protocol, input_pattern.initial_states)

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
                    msgs_list.append(msgs)
                    active_idx.append(node_idx)
                else:
                    sm.transition(protocol_related["state"].get_lost_state(), None)

            actions = get_actions(msgs_list, round_idx)
            for i, node_idx in enumerate(active_idx):
                smm.state_machines[node_idx].transition(actions[i], msgs_list[i])

        # Perform verification
        if not verifier.verify(protocol_related["protocol_class"], smm.state_machines):
            failed_cases.append(input_pattern)
        # optional: accumulate reward
        accumulate_reward += protocol_related["protocol_class"].get_reward(smm.state_machines)

    return failed_cases


def train_one_setting(protocol, nodes, rounds, model_name, invariance, training_params, protocol_related: dict, device='cpu'):
    # Initialization
    algo = importlib.import_module(f"model.{training_params["algo"]}")
    rig = initialize_input_generator(nodes, rounds, protocol_related)
    model = initialize_model(algo, model_name, nodes, rounds, protocol_related, device=device)
    if invariance == 0:
        rig.generate_all_inputs()
    else:
        rig.generate_filtered_inputs(invariance)
    env = Environment(
        state_class = protocol_related["state"],
        offset = protocol_related["state_offset"],
        device = device
    )
    smm = StateMachineManager(num_state_machines=nodes)
    sample_size = training_params["samples"]
    repetition = training_params["repetitions"]
    epochs = training_params["epochs"]
    sample_probability = config.SAMPLE_PROBABILITY
    failed_inputs = rig.all_inputs
    cur_epoch = 0
    success = False

    # Training
    while cur_epoch < epochs:
        cur_epoch += 1
        for _ in range(sample_size):
            input_pattern = sample_from_two_lists(rig.all_inputs, failed_inputs, sample_probability)
            train_one_case(protocol, rounds, algo, protocol_related, smm, env, model, input_pattern, repetition=repetition)

        def _get_actions(msgs_list: list, round_idx: int) -> list:
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions = model.get_greedy_action(state_tensor)
            next_states = env.step_all(actions)
            return next_states
        
        failed_inputs = evaluate_model(protocol, rounds, smm, protocol_related, _get_actions, failed_inputs)
        if len(failed_inputs) == 0:
            failed_inputs = evaluate_model(protocol, rounds, smm, protocol_related, _get_actions, rig.all_inputs)
            if len(failed_inputs) == 0:
                success = True
                break

    return success, cur_epoch


def train_one_case(protocol, rounds, algo, protocol_related, smm: StateMachineManager, env: Environment, model, input_pattern, repetition: int = 5):
    reward_list = []
    trajectories = dict()

    # Repeat rollout for the same input pattern 
    # (to reduce variance / increase efficiency of exploration)
    for _ in range(repetition):
        # Init state
        smm.initialize(protocol, input_pattern.initial_states)

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
                    msgs_list.append(msgs)
                    active_idx.append(node_idx)
                else:
                    sm.transition(protocol_related["state"].get_lost_state(), None)

            # Rollout
            state_tensor = env.get_state_all(msgs_list, round_idx)
            actions, extra = model.get_action(state_tensor)
            next_states = env.step_all(actions)

            # Transition each state machine
            for i, node_idx in enumerate(active_idx):
                smm.state_machines[node_idx].transition(next_states[i], msgs_list[i])

            algo.update_trajectories(trajectories, state_tensor, actions, extra)

        # Compute reward
        reward = protocol_related["protocol_class"].get_reward(smm.state_machines)
        algo.update_trajectories(trajectories)
        reward_list.append(reward)

    # Backpropagation
    algo.train_model(model, trajectories=trajectories, rewards=reward_list, others={"ppo_epochs": 10})


def run_benchmark_simple_majority(file_path: str = None):
    # Benchmark configuration
    repeat_times = 10
    nodes = [3, 4, 5]
    rounds = [1]
    model_and_invariance = [("mlp", 0), ("mlp_op", 3), ("set_transformer", 3)]
    training_params = {
        "epochs": 100,
        "samples": 100,
        "repetitions": 20,
        "algo": "a2c"
    }
    protocol = "simple_majority"
    protocol_related = PROTOCOL_TABLE[protocol]

    # Run benchmark
    results = []
    for bm_nodes, bm_rounds, (model_name, invariance) in product(nodes, rounds, model_and_invariance):
        setting_results = [
            f"Benchmarking {protocol} with {bm_nodes} nodes, {bm_rounds} rounds, model {model_name}, invariance {invariance}", # description
            0, # success count
            [], # used epochs list
            (protocol, bm_nodes, bm_rounds, model_name, invariance)
        ]
        for trial in range(repeat_times):
            logger.info(f"Trial {trial}, {setting_results[0]}")
            success, used_epochs = train_one_setting(
                protocol,
                bm_nodes,
                bm_rounds,
                model_name,
                invariance,
                training_params,
                protocol_related,
                device='cpu'
            )
            if success:
                setting_results[1] += 1
                setting_results[2].append(used_epochs)
                logger.info(f"Trial {trial} succeeded in {used_epochs} epochs.")
            else:
                logger.info(f"Trial {trial} failed after {used_epochs} epochs.")
        results.append(setting_results)
    
    # Print results
    for setting_result in results:
        description, success_count, used_epochs_list, _ = setting_result
        avg_epochs = sum(used_epochs_list) / len(used_epochs_list) if used_epochs_list else float('inf')
        logger.info(f"{description}: {success_count}/{repeat_times} succeeded, average epochs: {avg_epochs}")
    # Write results to file
    if file_path:
        with open(file_path, "wb") as f:
            pickle.dump(results, f)


def run_benchmark_atomic_commit(file_path: str = None):
    # Benchmark configuration
    repeat_times = 10
    nodes = [3, 4]
    rounds = [2]
    model_and_invariance = [("mlp", 0), ("mlp_op", 3), ("set_transformer", 3)]
    training_params = {
        "epochs": 100,
        "samples": 100,
        "repetitions": 20,
        "algo": "dqn"
    }
    protocol = "atomic_commit"
    protocol_related = PROTOCOL_TABLE[protocol]

    # Run benchmark
    results = []
    for bm_nodes, bm_rounds, (model_name, invariance) in product(nodes, rounds, model_and_invariance):
        setting_results = [
            f"Benchmarking {protocol} with {bm_nodes} nodes, {bm_rounds} rounds, model {model_name}, invariance {invariance}", # description
            0, # success count
            [], # used epochs list
            (protocol, bm_nodes, bm_rounds, model_name, invariance)
        ]
        for trial in range(repeat_times):
            logger.info(f"Trial {trial}, {setting_results[0]}")
            success, used_epochs = train_one_setting(
                protocol,
                bm_nodes,
                bm_rounds,
                model_name,
                invariance,
                training_params,
                protocol_related,
                device='cpu'
            )
            if success:
                setting_results[1] += 1
                setting_results[2].append(used_epochs)
                logger.info(f"Trial {trial} succeeded in {used_epochs} epochs.")
            else:
                logger.info(f"Trial {trial} failed after {used_epochs} epochs.")
        results.append(setting_results)
    
    # Print results
    for setting_result in results:
        description, success_count, used_epochs_list, _ = setting_result
        avg_epochs = sum(used_epochs_list) / len(used_epochs_list) if used_epochs_list else float('inf')
        logger.info(f"{description}: {success_count}/{repeat_times} succeeded, average epochs: {avg_epochs}")
    # Write results to file
    if file_path:
        with open(file_path, "wb") as f:
            pickle.dump(results, f)


def run_benchmark_dataset():
    protocol = "simple_majority"
    players = [3, 4, 5, 6]
    rounds = [1]
    invariance = [0, 3]
    for p, r, inv in product(players, rounds, invariance):
        rig = initialize_input_generator(p, r, PROTOCOL_TABLE[protocol])
        if inv == 0:
            rig.generate_all_inputs()
        else:
            rig.generate_filtered_inputs(inv)
        logger.info("Generated {} inputs for protocol {}, players {}, rounds {}, invariance {}".format(
            len(rig.all_inputs), protocol, p, r, inv
        ))
    
    protocol = "atomic_commit"
    players = [3, 4, 5, 6]
    rounds = [2, 3]
    invariance = [0, 3]
    for p, r, inv in product(players, rounds, invariance):
        rig = initialize_input_generator(p, r, PROTOCOL_TABLE[protocol])
        if inv == 0:
            rig.generate_all_inputs()
        else:
            rig.generate_filtered_inputs(inv)
        logger.info("Generated {} inputs for protocol {}, players {}, rounds {}, invariance {}".format(
            len(rig.all_inputs), protocol, p, r, inv
        ))


if __name__ == "__main__":
    import sys
    option = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if option == 1:
        run_benchmark_simple_majority(file_path="benchmark_sm.pkl")
    elif option == 2:
        run_benchmark_atomic_commit(file_path="benchmark_ac.pkl")
    elif option == 3:
        run_benchmark_dataset()
