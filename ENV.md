# Environment and Command

## Environment

Python: 3.11+ (Exact match: 3.12.7)

Remember to pull the submodule:

`git submodule update --init --recursive`

Set virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install requirements:

```bash
pip install -r requirements.txt
```

## How to run

Most commands I use are listed in a [shell script](./run.sh)

You can check the usage and all supported options by `python main.py --help`.

**If you want to save the model into a directory, make sure the directory exists.**

## Example

To test:

`python -m pytest -s`

To train a model:

`python -u main.py -p 4 -r 1 -P simple_majority --algorithm dqn --model mlp_op --model_save temp.pth --log_level info > temp.log 2>&1`

- `-p`: the number of players.
- `-r`: the number of rounds.
- `-P`: the name of the protocol.
- `--algorithm`: RL algorithm.
- `--model`: underlying model.

To evaluate the results:

`python main.py -p 3 -r 1 -P simple_majority --evaluate --algorithm dqn --model set_transformer --model_load results/models/simple_majority/dqn_st_3_1.pth > eva_ext_dqn_st_p3_1_r1.log 2>&1`

- `--model_load`: make sure you use `-gt` or `--model_load` to provide a policy to choose actions.
- **You should ensure that the model you loaded fits `-p`, `-r`, `--algorithm` and `--model`.**

To evaluate if the model is generalizable:

`python main.py -p 3 -r 1 -ep 1 -P simple_majority --evaluate --algorithm dqn --model set_transformer --model_load results/models/simple_majority/dqn_st_3_1.pth > eva_ext_dqn_st_p3_1_r1.log 2>&1`

- `-ep`: extended players. Ensure `-p` and `-r` are same as the model setting, you can extend it to any number of players. But too large may cause a long verification.

## Benchmark

```bash
python -u benchmark.py [option] benchmark.log 2>&1
```

option:

- 1: train on simple majority protocol with 3-5 nodes, 1 round, 3 models, a2c algorithm. Repeat 10 times.
- 2: train on atomic commit protocol with 3-4 nodes, 2 round, 3 models, dqn algorithm. Repeat 10 times.
- 3: statistics of size of datasets, 3-6 nodes 1 round for simple majority, 3-6 nodes 2-3 rounds for atomic commit.

Invariance level are set to 0 (full size) and 3 (highest reduced).
