from grid_runner.runner import GridRunner

run = GridRunner("example_run")
args = ["hello", "world"]
lr = ["1e-3", "1e-4"]

template = "nohup python example/some_train_code.py --name {} --lr {}"
train_args = [args, lr]

train_instruction = run.gen_instruction(
    template, train_args, suffix="train")

run.run(["example_train"], train_instruction, gpus=[4, 5])
