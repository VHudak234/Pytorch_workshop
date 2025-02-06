#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/agnews'
base_call = (f"python main_opacus.py --i {DATA_HOME} --o {DATA_HOME}/output --dataset 'agnews' --epochs 20 ")

learning_rates = [0.00005,0.00001,0.000025]
epsilons = [2,5,8,12,15,20]

settings = [(lr, eps) for lr in learning_rates for eps in epsilons]
nr_expts = len(learning_rates) * len(epsilons)

nr_servers = 10
avg_expt_time = 100  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("TextEps2_20_Lr5e5_to_1e5.txt", "w")

for lr, eps in settings:
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--epsilon {eps} "
    )
    print(expt_call, file=output_file)

output_file.close()