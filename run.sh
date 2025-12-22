#!/bin/sh

# pytest
# python -m pytest -s

################ evaluation test groundtruth
# python main.py -p 3 -r 1 -P simple_majority -gt --evaluate > evaluate.log 2>&1
# python main.py -p 3 -r 3 -P atomic_commit -gt --evaluate --log_level info > gt.log 2>&1
# python main.py -p 5 -r 3 -P primary_backup -gt --evaluate --log_level info > gt.log 2>&1

################ evaluation test trained models
# python main.py -p 3 -r 1 -ep 1 -P simple_majority --evaluate --algorithm dqn --model set_transformer --model_load results/models/simple_majority/dqn_st_3_1.pth > eva_ext_dqn_st_p3_1_r1.log 2>&1
# python main.py -p 3 -r 3 -P atomic_commit --evaluate --algorithm dqn --model set_transformer --model_load results/models/atomic_commit/dqn_st_3_3.pth --log_level info > evaluate.log 2>&1
# python main.py -p 3 -r 4 -ep 1 -P primary_backup --evaluate --algorithm dqn --model set_transformer --model_load results/temp/dqn_st_3_3.pth > evaluate.log 2>&1
# python main.py -p 3 -r 2 -P primary_backup --evaluate --algorithm dqn --model set_transformer --model_load results/models/primary_backup/dqn_st_4_2.pth > evaluate.log 2>&1

################ train test simple majority
# python -u main.py -p 3 -r 1 -P simple_majority --algorithm a2c --model set_transformer --model_save results/temp/temp.pth --log_level info > temp.log 2>&1
# python -u main.py -p 5 -r 1 -P simple_majority --algorithm dqn --model set_transformer --model_save results/temp/st_5_1.pth --log_level info > temp.log 2>&1

################ train test atomic commit
# python -u main.py -p 3 -r 3 -P atomic_commit --algorithm a2c --model mlp_op --model_save results/temp/temp.pth --epochs 100 --log_level info > temp.log 2>&1
# python -u main.py -p 4 -r 2 -P atomic_commit --algorithm dqn --model set_transformer --model_save results/temp/dqn_st_4_3.pth --epochs 200 --log_level info > dqn_st_p4_r3.log 2>&1

################ train test primary backup
# python -u main.py -p 3 -r 3 -P primary_backup --algorithm a2c --model mlp_op --model_save results/temp/temp.pth --epochs 100 --log_level info > temp.log 2>&1
# python -u main.py -p 3 -r 3 -P primary_backup --algorithm dqn --model set_transformer --model_save results/temp/dqn_st_3_3.pth --epochs 200 --log_level info > dqn_st_p3_r3.log 2>&1
