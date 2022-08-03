# -*- coding : utf-8 -*-
# @FileName  : run_exp.py
# @Author    : Ruixiang JIANG (Songrise)
# @Github    : https://github.com/songrise
# @Description: Flyweight tool to manage shell commands.

import time
from typing import List, Optional
import numpy as np
import os
import datetime
import json


class Runner():
    """helper class to execute sh scripts"""

    def __init__(self, log_name=None, log_main: bool = True) -> None:
        """
        Initialize the runner.

        Parameters:
        ----------
        log_name : str, the name of the grid experiment.
        log_main : bool, if True, write log to main log file.
        """
        self.log_root = log_name
        # try to create log folder
        if self.log_root == None:
            # use crt time as log name
            self.log_root = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        try:
            os.mkdir(os.path.join(os.getcwd(), 'log'))
        except FileExistsError:
            pass
        try:
            os.mkdir(os.path.join(os.getcwd(), 'log', self.log_root))
        except FileExistsError:
            pass

        self.log_root = os.path.join(os.getcwd(), 'log', self.log_root)

        # create main log file
        self.log_main = log_main
        if log_main:
            with open(os.path.join(self.log_root, "main.txt"), 'w') as f:
                f.write("")
                f.close()

            self.main_log = os.path.join(self.log_root, "main.txt")

    def run(self, exp_names: List[str], instructions: List[str], gpus: List[int] = None, interval_time: int = 0, **kwargs: dict) -> None:
        """
        execute experiments in parallel, save log to log_name

        Parameters:
        ----------
        exp_name : List[str], the name for each experiment
        instructions : List[str], the command line instruction for each experiment
        gpus : List[int], the gpu id for each experiment, if None, assume use CUDA 0.
        interval_time : int, interval time (in minutes) between experiments. 
                    Only useful when number of experiments is larger than avaliable gpu.
                    Notice, it assume the gpu is available after the previous experiment.
        """
        if interval_time < 0:
            raise ValueError("interval time must be positive!")

        if len(exp_names) != len(instructions):
            self.log(
                "The number of experiment names and instructions does not match!")
            self.log("Fallback to use default experiment name")
            exp_names = [f"exp_{i}" for i in range(len(instructions))]

        if exp_names is None:
            self.log("No experiment name specified, use default name!")
            exp_names = [f"exp_{i}" for i in range(len(instructions))]

        self.log("Starting experiments with gpus: {}".format(gpus))
        self.log("Experiments instruction: {}".format(instructions))

        # excecute experiments in parallel
        if gpus is None:
            self.log("No gpus specified, using CUDA_0 by default!")
            gpus = [0]

        num_gpu = len(gpus)

        for i, ins in enumerate(instructions):
            if i != 0 and i % num_gpu == 0:
                self.log(
                    "Current running experiment is {}/{}".format(i, len(instructions)))
                os.system(f"ps >> {self.main_log}")
                self.log(
                    "Waiting {} minutes for next set of experiments".format(interval_time))
                time.sleep(interval_time*60)
                self.log(
                    "{} minutes past, attempt to start new set of experiments".format(interval_time))

            self.log("Experiment {}: {}".format(i, ins))
            redirect = "> {}_gpu{}.out".format(
                self.log_root+"/"+exp_names[i], gpus[i % num_gpu])

            os.system("CUDA_VISIBLE_DEVICES={} {} {} &".format(
                gpus[i % num_gpu], instructions[i], redirect))

        # get process id
        os.system(f"ps >> {self.main_log}")

    def compose(self, template: str, args: List[List], dump_param: Optional[bool] = True, suffix: Optional[str] = None) -> List[str]:
        """
        Generate a list of instruction from template and args.

        Parameters:
        ----------
        template : str, the template of instruction in the form of format string, specify the blank as {}.
        args : List[List], the list of arguments for each instruction, must much the number of {} in template.
        dump_param : bool, Optional. if True, dump the args and template to a file.
        suffix : str, the suffix for the dumped file.

        Returns:
        ----------
        instructions : list, the list of instructions that are ready to run.
        """
        # log template and args
        if dump_param:
            try:
                os.mkdir(os.path.join(self.log_root, 'param'))
            except FileExistsError:
                pass
            with open(os.path.join(self.log_root, f"param/args_{suffix}.json"), "w") as f:
                # dump as json
                json.dump(args, f)
            with open(os.path.join(self.log_root, f"param/template_{suffix}.json"), "w") as f:
                json.dump(template, f)

        # reshape for list unpacking
        args = np.array(args)
        args = args.T
        args = args.tolist()
        return [template.format(*a) for a in args]

    def log(self, content: str) -> None:
        """
        Write log to the main log file.
        """
        if self.log_main:
            with open(self.main_log, 'a') as f:
                f.write(datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")+"\t|"+content)
                f.write('\n')
                f.close()
        return

    def load(self, file_dir: str):
        """
        Load dumped args or templates from file.
        Parameters:
        ----------
        file_dir: str, the path to the json file.

        Returns:
        ----------
        content: list or str, the content of the file.
        """
        with open(os.path.join(self.log_root, file_dir), 'r') as f:
            content = json.load(f)
        return content

    def __str__(self) -> str:
        return "Runner at: {}".format(self.log_root)
