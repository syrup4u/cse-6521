#!/bin/sh

# pytest
# python -m pytest -s

# must initialize a workdir at first
# python main.py init --workdir results/temp/test

################ evaluation test groundtruth
## --invariance 3 --log_level debug
python main.py evaluate -p 4 -r 2 -gt --workdir results/temp/test2

################ evaluation test trained models
# python main.py evaluate -p 11 -r 2 --invariance 3 --workdir results/temp/test2

################ training
# python -u main.py train -p 3 -r 2 --workdir results/temp/test2

################ generalization test (evaluate groundtruth and human pooling)
# python generalize.py evaluate -p 4 -r 3 -P atomic_commit --gt_l1 --human_l2 --log_level debug > gt.log 2>&1
