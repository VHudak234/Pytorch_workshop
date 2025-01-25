#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/cifar10'
base_call = (f"python main_opacus.py -i {DATA_HOME}/input -o {DATA_HOME}/output --Dice 1 ")

repeats = 2
learning_rates = [0.05, 0.01]
epsilons = [2,10]
clipping_norms = [1,5,10]

settings = [(lr, eps, clipping_norm) for lr in learning_rates for eps in epsilons for clipping_norm in clipping_norms
            for repeat in range(repeats)]
nr_expts = len(learning_rates) * len(epsilons) * 2

nr_servers = 10
avg_expt_time = 100  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("DPSGD_EPS2-10_LR0.05-0.01-Dice.txt", "w")

for lr, eps, clipping_norm in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--epsilon {eps} "
        f"--max-grad-norm {clipping_norm} "
        f"--max-ef-norm {clipping_norm} "
    )
    print(expt_call, file=output_file)

output_file.close()