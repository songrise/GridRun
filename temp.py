# -*- coding : utf-8 -*-
# @FileName  : run_exp.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jul 25, 2022
# @Github    : https://github.com/songrise
# @Description: Run experiments for StyleHyperNeRF

from typing import List
import numpy as np
import os
import datetime
import subprocess

class Runner():
    """helper class to execute sh scripts"""
    def __init__(self, exp_name=None, log_main:bool=True) -> None:
        self.log_root = exp_name
        #try to create log folder
        if self.log_root == None:
            #use crt time as log name
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

        #create main log file
        self.log_main = log_main
        if log_main:
            with open(os.path.join(self.log_root, "main.txt"),'w') as f:
                f.write("")
                f.close()

            self.main_log = os.path.join(self.log_root, "main.txt")


    def run(self,exp_name:List[str],instruction:List[str], gpus:List[int],end_time:int = -1,**kwargs:dict):
        """
        execute experiments in parallel, save log to log_name
        """
        self.log("Starting experiments with gpus: {}".format(gpus))
        self.log("Experiments instruction: {}".format(instruction))

        #! for experiment on cityu server
        for _ in range(4):
            assert _ not in gpus
        #excecute experiments in parallel
        
        for i,gpu in enumerate(gpus):
            self.log("Experiment {}: {}".format(i,instruction[i]))
            redirect = "> {}_gpu{}.out".format(self.log_root+"/"+exp_name[i],gpu)
            os.system("CUDA_VISIBLE_DEVICES={} {} {} &".format(gpu,instruction[i],redirect))
        #get process id
        os.system(f"ps >> {self.main_log}")
            


    def gen_instruction(self,template, args:List[List],dump_param:bool=True, suffix:str= None) -> list:
        #log template and args
        if dump_param:
            with open(os.path.join(self.log_root,f"args_{suffix}.txt"),"a") as f:
                f.write(str(args))
            with open(os.path.join(self.log_root,f"template_{suffix}.txt"),"a") as f:
                f.write(template)

        #reshape for list unpacking
        args = np.array(args)
        args = args.T
        args = args.tolist()
        return [template.format(*a) for a in args]

    def log(self, content:str):
        if self.log_main:
            with open(self.main_log,'a') as f:
                f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"\t|"+content)
                f.write('\n')
                f.close()
        return 

    def load_template(self,path:str):
        with open(path,'r') as f:
            template = f.read()
        return str(template)
    
    def load_args(self,path:str):
        with open(path,'r') as f:
            args = f.read()
        return str(args)

if __name__ == '__main__':
    # suffix = ["lord", "superman","modi", "vincent"]
    # ttext = ["Lord Voldermort", "Superman cartoon", "Modigliani", "Vincent van Gogh portrait"]
    suffix = ["yanan_nature","fangzhou_nature"]
    # it is also possible to load template from a file
    train_template = "nohup python exp_runner.py --mode train --conf ./confs/womask_{}.conf --case {}"
    
    eval_template = "nohup python eval.py  --scene_name fangzhou_nature_replicate_2_{}_ALLFF \
            --gif_name fangzhou_nature_replicate_2_{}_ALLFF  --dataset_name llff \
            --img_wh 270 480  --N_importance 128   --N_emb_dir 4 \
            --coarse_path /home/wangcan/ruixiang/StyleHyperNeRF/ckpts/fangzhou_nature/{}/nerf_coarse_{}_epoch_5_step_90.pt \
            --fine_path /home/wangcan/ruixiang/StyleHyperNeRF/ckpts/fangzhou_nature/{}/nerf_fine_{}_epoch_5_step_90.pt\
            --root_dir datasets/nerf_llff_data/fangzhou/fangzhou_nature"
    train_args = [suffix, suffix]
    eval_args = [suffix, suffix, suffix, suffix, suffix, suffix]


    runner = Runner(exp_name="no_undistort")
    # print(runner.load_template("/home/wangcan/ruixiang/StyleHyperNeRF/log/replicate_fangzhou/template.txt"))
    # print(runner.load_args("/home/wangcan/ruixiang/StyleHyperNeRF/log/replicate_fangzhou/args_eval.txt"))
    exp_name_train = ["fangzhou_nature_{}_train".format(s) for s in suffix] #nohup suffix
    exp_name_eval = ["fangzhou_nature_{}_eval".format(s) for s in suffix] #nohup suffix
    train_instruction = runner.gen_instruction(train_template, train_args, suffix="train")
    # eval_instruction = runner.gen_instruction(eval_template,eval_args,"eval")
    # runner.run(exp_name_train,train_instruction,gpus=[4,5,6,7])
    runner.run(exp_name_train,train_instruction,[4,5]) 
