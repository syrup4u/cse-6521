import config
from process import general

import argparse
import logging
import os

# Set up logging configuration
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(prog="Learn Protocols")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--workdir", type=str, required=True, help="working directory for config, logs and models")
    parent_parser.add_argument("--log_level", type=str, default="info", help="logging level", choices=["debug", "info", "warning", "error", "critical"])

    parser_init = subparsers.add_parser("init", help="initialize working dir and config (must do)", parents=[parent_parser])
    parser_init.set_defaults(func=initialize)

    parent_parser.add_argument("--players", "-p", type=int, required=True, help="number of players")
    parent_parser.add_argument("--rounds", "-r", type=int, required=True, help="number of rounds")

    parser_train = subparsers.add_parser("train", help="train the model to learn a protocol", parents=[parent_parser])
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate the trained model on a protocol", parents=[parent_parser])
    parser_evaluate.add_argument("--groundtruth", "-gt", action='store_true', help="use ground truth (human designed) for evaluation")
    parser_evaluate.add_argument("--invariance", type=int, default=0, help="input invariance level for filtering input patterns (0-3)")
    parser_evaluate.add_argument("--z3", action='store_true', help="use z3 verifier for evaluation")
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    return args

def set_logger(fp):
    log_handler = logging.FileHandler(fp, mode='w')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)


def initialize(args):
    os.makedirs(args.workdir, exist_ok=True)
    config_fp = os.path.join(args.workdir, config.CONFIG_PATH)
    if not os.path.exists(config_fp):
        config.create_default_config(config_fp)
        logger.info(f"Created default config at {config_fp}.")
    else:
        logger.info(f"Config file already exists at {config_fp}.")

# TODO: multiprocessing and sampling for evaluation
def evaluate(args):
    assert os.path.exists(args.workdir), "Should have initialized the working directory first."
    cfg: config.Config = config.load_config(os.path.join(args.workdir, config.CONFIG_PATH))
    set_logger(os.path.join(args.workdir, config.EVAL_LOG_PATH))

    general.evaluate(cfg, args.rounds, args.players, others={
        "use_groundtruth": args.groundtruth,
        "invariance_level": args.invariance,
        "model_path": os.path.join(args.workdir, config.MODEL_PATH),
        "use_z3": args.z3
    })

def train(args):
    assert os.path.exists(args.workdir), "Should have initialized the working directory first."
    cfg = config.load_config(os.path.join(args.workdir, config.CONFIG_PATH))
    set_logger(os.path.join(args.workdir, config.TRAIN_LOG_PATH))

    general.train(cfg, args.rounds, args.players, others={
        "model_path": os.path.join(args.workdir, config.MODEL_PATH)
    })


def main():
    args = parse_args()
    logger.setLevel(args.log_level.upper())
    args.func(args)

if __name__ == "__main__":
    main()
