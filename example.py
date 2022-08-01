from runner import GridRunner

runner = GridRunner("example_run")
args = ["hello", "world"]
lr = ["1e-3", "1e-4"]

template = "nohup python example/some_train_code.py --name {} --lr {}"
train_args = [args, lr]

train_instruction = runner.gen_instruction(
    template, train_args, suffix="train")

runner.run(["example_train"], train_instruction, gpus=[4, 5])
