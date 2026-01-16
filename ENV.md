# Environment and Command

## Environment

Python: 3.10+ (Exact match: 3.12.7)

Remember to pull the submodule:

`git submodule update --init --recursive`

Set virtual environment:

```bash
python -m venv .venv
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

**Remember to initialize a workdir first by `python main.py init --workdir <your_work_dir>`**

## Hyper Parameters

Hyper parameters for training are defined in the config file in workdir. For reference: [default config](./default_config.yaml).

## Model Saving

The model will be saved automatically in workdir named `model.pth` when training is complete. If you want to start training a model from scratch rather than continuing from the checkpoint, make sure to remove or rename the file `model.pth`.

For evaluation, make sure the `model.pth` exists or you use `-gt` for groundtruth.

## Example

To test:

`python -m pytest -s`

To initialize a new workflow:

`python main.py init --workdir results/temp/ac_p3_r3`

To train a model:

`python -u main.py train -p 3 -r 3 --workdir results/temp/ac_p3_r3`

- `-p`: the number of players.
- `-r`: the number of rounds.

To evaluate the model (2 ways):

`python main.py evaluate -p 8 -r 8 --z3 --workdir results/temp/ac_p3_r3`
`python main.py evaluate -p 5 -r 4 --workdir results/temp/ac_p3_r3`

- Recommended: enable `--z3` which will use z3 verifier to find counter examples. This is much faster than the default method.
- Default: exhaust all possible input patterns. `--invariance 3` can reduce the size. If not using `--invariance 3`, never exceed 5 players and 5 rounds, it will explode.
- You don't need to set `-p` and `-r` same as when for training, which is helpful for verifying if the model is generalizable.

## Benchmark

```bash
python -u benchmark.py [option] benchmark.log 2>&1
```

option:

- 1: train on simple majority protocol with 3-5 nodes, 1 round, 3 models, a2c algorithm. Repeat 10 times.
- 2: train on atomic commit protocol with 3-4 nodes, 2 round, 3 models, dqn algorithm. Repeat 10 times.
- 3: statistics of size of datasets, 3-6 nodes 1 round for simple majority, 3-6 nodes 2-3 rounds for atomic commit.
- 4: a custom setting for quick test.

Invariance level are set to 0 (full size) and 3 (highest reduced).
