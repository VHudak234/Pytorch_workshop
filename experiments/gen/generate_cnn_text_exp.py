DATA_HOME = "/Users/vincehudak/Documents/Intellij Projects/Pytorch_workshop/yelp"

script_root = "/Users/vincehudak/Documents/Intellij Projects/Pytorch_workshop/fnn_text.py"

base_call = f'python "{script_root}" -i "{DATA_HOME}/input" -o "{DATA_HOME}/output" --epochs 15 --private --model cnn'

learning_rates = [0.0001, 0.0005, 0.00005]
epsilons = [2,8,20]

settings = [(lr, eps) for lr in learning_rates for eps in epsilons]

output_file = open('CNN_DPSGDText_Experiment.txt', 'w')

for lr, eps in settings:
    expt_call = (
        f"{base_call} "
        f"--lr {lr} "
        f"--epsilon {eps}"
    )
    print(expt_call, file=output_file)

output_file.close()