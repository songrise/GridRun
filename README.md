# GridRun
GridRun is a simple tool that helps you to execute and manage similar shell commands. Originally, it is designed to perform grid-search of hyper-parameter in deep learning experiments. However, it can be used for other purposes as well. 


## Installation
install via [pip](https://pip.pypa.io/en/stable/installing/)
```
pip install grid_run
```

## Usage
When you want to execute similar shell commands on a multi-gpu machine, you can use GridRun to generate sh commands and run in parallel. 

Consider a simple use case, you want to find the best learning rate, you can use the following command:

```Python
from grid_run.runner import Runner

runner = Runner(log_name="lr_search")

exp_name = ["1e-3","1e-2","1e-1","1","10","100"]
lr = [0.001, 0.01, 0.1, 1, 10, 100]

# Do not write & to the end of the command, it will be added automatically.
template = "nohup python -u train.py --exp {} --lr {}" 

train_instructions = runner.gen_instruction(template,[exp_name,lr])
gpus = [0,1] 
runner.run(train_instructions,gpus = gpus)

# if you have less gpu, you can calculate the running time per job (e.g., 10 min), and run in sequence.
runner.run(train_instructions,gpus = [0], interval_time = 10)
```

Grid runner will log the running status as well as hparams to log directory. For example:
```
.
└── example_run
    ├── exp_0_gpu4.out
    ├── exp_1_gpu5.out
    ├── main.txt
    └── param
        ├── args_train.json
        └── template_train.json
```

This allows you to easily analyze the running status for each of hte experiments, as well as reproducing the experiment.

## Others
GridRun is written in a extremely simple way (~100 lines). It is originally a custom script that I write in a hour, after finding it is tedious to execute batches of jobs manually. I will add more features later. Also, if you find any bugs or have any suggestions, please open an issue.
