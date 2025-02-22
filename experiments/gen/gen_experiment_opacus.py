#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/cifar10'
base_call = (f"python main_opacus.py -i {DATA_HOME}/input -o {DATA_HOME}/output ")

epochs = [20,15]
learning_rates = [0.1, 0.05, 0.01, 0.001]
epsilons = [8,12,15]

settings = [(lr, eps, epoch) for lr in learning_rates for eps in epsilons
            for epoch in epochs]
nr_expts = len(learning_rates) * len(epsilons) * 2

nr_servers = 10
avg_expt_time = 100  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("experimentEps8_12_15_Lr0.1_to_0.0001.txt", "w")

for lr, eps, epoch in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--epsilon {eps} "
        f"--epochs {epoch}"
    )
    print(expt_call, file=output_file)

output_file.close()