#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/imdb_data'

dp_base_call = f'python fnn_imdb.py -i "{DATA_HOME}/input" -o "{DATA_HOME}/output" --epochs 25 --private'

dp_base_call_scheduler = f'python fnn_imdb.py -i "{DATA_HOME}/input" -o "{DATA_HOME}/output" --epochs 25 --private --scheduler'

sgd_base_call = f'python fnn_imdb.py -i "{DATA_HOME}/input" -o "{DATA_HOME}/output" --epochs 25'

learning_rates = [0.0001, 0.0005, 0.00005]
epsilons = [2,5,10,20]
settings = [(lr, eps) for lr in learning_rates for eps in epsilons]

sch_learning_rates = [0.0001, 0.0005, 0.00005]
sch_epsilons = [2,5,10,20]
sch_settings = [(lr, eps) for lr in sch_learning_rates for eps in sch_epsilons]

output_file = open('FNNImdb_Text_experiments.txt', 'w')

for lr, eps in settings:
    expt_call = (
        f"{dp_base_call} "
        f"--lr {lr} "
        f"--epsilon {eps}"
    )
    print(expt_call, file=output_file)

for lr, eps in sch_settings:
    expt_call = (
        f"{dp_base_call_scheduler} "
        f"--lr {lr} "
        f"--epsilon {eps}"
    )
    print(expt_call, file=output_file)

for lr in learning_rates:
    expt_call = (
        f"{sgd_base_call} "
        f"--lr {lr}"
    )
    print(expt_call, file=output_file)


output_file.close()