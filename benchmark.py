"""
This is a benchmark for evaluating the effectiveness and performance of
the whole framework, including various algorithms / models / methods.
"""
import config
import common
from process import general

import logging
from itertools import product
import pickle

LOG_PATH="benchmark.log"
CONFIG_PATH="default_config.yaml"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
log_handler = logging.FileHandler(LOG_PATH, mode='w')
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)
cfg: config.Config = config.load_config(CONFIG_PATH)

def run_benchmark_simple_majority(file_path: str = None):
    # Benchmark configuration
    repeat_times = 10
    nodes = [3, 4, 5]
    rounds = [1]
    model_and_invariance = [("mlp", 0), ("mlp_op", 3), ("set_transformer", 3)]
    cfg.algorithm.name = "a2c"
    cfg.protocol = "simple_majority"

    # Run benchmark
    results = []
    for bm_nodes, bm_rounds, (model_name, invariance) in product(nodes, rounds, model_and_invariance):
        setting_results = [
            f"Benchmarking {cfg.protocol} with {bm_nodes} nodes, {bm_rounds} rounds, model {model_name}, invariance {invariance}", # description
            0, # success count
            [], # used epochs list
            (cfg.protocol, bm_nodes, bm_rounds, model_name, invariance)
        ]
        for trial in range(repeat_times):
            logger.info(f"Trial {trial}, {setting_results[0]}")
            cfg.model.name = model_name
            cfg.train.invariance_level = invariance
            success, used_epochs = general.train(cfg, bm_rounds, bm_nodes)
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


# TODO: multiprocessing for faster benchmarking
def run_benchmark_atomic_commit(file_path: str = None):
    # Benchmark configuration
    repeat_times = 10
    nodes = [3, 4]
    rounds = [2]
    model_and_invariance = [("mlp", 0), ("mlp_op", 3), ("set_transformer", 3)]
    cfg.algorithm.name = "dqn"
    cfg.protocol = "atomic_commit"

    # Run benchmark
    results = []
    for bm_nodes, bm_rounds, (model_name, invariance) in product(nodes, rounds, model_and_invariance):
        setting_results = [
            f"Benchmarking {cfg.protocol} with {bm_nodes} nodes, {bm_rounds} rounds, model {model_name}, invariance {invariance}", # description
            0, # success count
            [], # used epochs list
            (cfg.protocol, bm_nodes, bm_rounds, model_name, invariance)
        ]
        for trial in range(repeat_times):
            logger.info(f"Trial {trial}, {setting_results[0]}")
            cfg.model.name = model_name
            cfg.train.invariance_level = invariance
            success, used_epochs = general.train(cfg, bm_rounds, bm_nodes)
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
    protocol_related = common.PROTOCOL_TABLE[protocol]
    for p, r, inv in product(players, rounds, invariance):
        rig = common.initialize_input_generator(protocol_related, p, r, inv)
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
    protocol_related = common.PROTOCOL_TABLE[protocol]
    for p, r, inv in product(players, rounds, invariance):
        rig = common.initialize_input_generator(protocol_related, p, r, inv)
        if inv == 0:
            rig.generate_all_inputs()
        else:
            rig.generate_filtered_inputs(inv)
        logger.info("Generated {} inputs for protocol {}, players {}, rounds {}, invariance {}".format(
            len(rig.all_inputs), protocol, p, r, inv
        ))

def run_benchmark_dataset_custom():
    import time
    protocol = "atomic_commit"
    players = [7]
    rounds = [3]
    invariance = [0, 3]
    protocol_related = common.PROTOCOL_TABLE[protocol]
    for p, r, inv in product(players, rounds, invariance):
        rig = common.initialize_input_generator(protocol_related, p, r, inv)
        if inv == 0:
            start = time.perf_counter()
            rig.generate_all_inputs()
            end = time.perf_counter()
            logger.info("Time taken to generate all inputs: {:.2f} seconds".format(end - start))
        else:
            start = time.perf_counter()
            rig.generate_filtered_inputs(inv)
            end = time.perf_counter()
            logger.info("Time taken to generate filtered inputs with invariance {}: {:.2f} seconds".format(inv, end - start))
            logger.info("Generated {} inputs for protocol {}, players {}, rounds {}, invariance {}".format(len(rig.all_inputs), protocol, p, r, inv))


if __name__ == "__main__":
    import sys
    option = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    if option == 1:
        run_benchmark_simple_majority(file_path="benchmark_sm.pkl")
    elif option == 2:
        run_benchmark_atomic_commit(file_path="benchmark_ac.pkl")
    elif option == 3:
        run_benchmark_dataset()
    elif option == 4:
        run_benchmark_dataset_custom()
