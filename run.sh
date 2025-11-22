#!/bin/sh

# simple test
# python main.py -p 4 -r 1 -P simple_majority -gt --evaluate
# python main.py -p 4 -r 3 -P atomic_commit -gt --evaluate > gt2.log 2>&1

# debug test
# python main.py -p 4 -r 1 -P simple_majority -gt --evaluate --log_level debug > temp.log 2>&1

# train test
# python main.py -p 3 -r 1 -P simple_majority --model mlp_op --algorithm a2c --log_level debug > temp.log 2>&1
python main.py -p 3 -r 1 -P simple_majority --model set_transformer --algorithm a2c --log_level debug > temp.log 2>&1
