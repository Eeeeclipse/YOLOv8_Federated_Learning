# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import shutil
import cv2
import random
import matplotlib.pyplot as plt
import copy
import wandb

from ultralytics import YOLO

import torch



def average_models(models):
    model_params = [model.model.state_dict() for model in models]
    averaged_params = {}
    for param_name in model_params[0]:
        params = torch.stack([model_params[i][param_name] for i in range(len(models))])
        params = params.to(torch.float16)
        averaged_params[param_name] = torch.mean(params, dim=0)
    return averaged_params



num_local_step = 1
num_comm = 20
num_client = 4
lr = 0.0000001

curr_path=os.getcwd()
config_path = os.path.join(curr_path, 'config.yaml')

config_paths = []
config_path1 = os.path.join(curr_path, 'client1.yaml')
config_paths.append(config_path1)

config_path2 = os.path.join(curr_path, 'client2.yaml')
config_paths.append(config_path2)

config_path3 = os.path.join(curr_path, 'client3.yaml')
config_paths.append(config_path3)

config_path4 = os.path.join(curr_path, 'client4.yaml')
config_paths.append(config_path4)

import os
os.environ['WANDB_DISABLED'] = 'true'

if __name__ == '__main__':
    model = YOLO('./yolov8n.yaml').load('yolov8n.pt')

    for _ in range(num_comm):
        print("-----------------------------------------------------------------------------")
        print('num_comm: ', _)
        print("-----------------------------------------------------------------------------")
        models = [copy.deepcopy(model) for _ in range(num_client)]
        for index, dup_model in enumerate(models):
            dup_model.train(data=config_paths[index], epochs=num_local_step, batch=16, save=True, resume=True, iou=0.5, conf=0.001, plots=False, workers = 0, lr0 = lr, lrf = lr)
        ensemble_model_params = average_models(models)
        model.model.load_state_dict(ensemble_model_params)
        name = 'FedAvg_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, name)

    torch.save(model,'final_model.pt')
