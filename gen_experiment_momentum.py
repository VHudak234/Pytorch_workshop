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

# learning_rates = [0.05, 0.01]
epsilons = [2,20]
momentums = [0.7, 0.9]

settings = [(eps, momentum)
            # for lr in learning_rates
            for eps in epsilons
            for momentum in momentums]
nr_expts = len(momentums) * len(epsilons) * 2

nr_servers = 10
avg_expt_time = 100  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("DPSGD_EPS2-20Momentum0.7-0.9BatchSize32.txt", "w")

for eps, momentum in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--batch-size 32 "
        f"--epsilon {eps} "
        f"--momentum {momentum}"
    )
    print(expt_call, file=output_file)

output_file.close()