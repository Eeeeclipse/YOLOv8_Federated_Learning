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

def get_grads_(model, server_model, lr):
    grads = []
    server_model_params = server_model.model.state_dict()
    model_params = model.model.state_dict()
    for param_name in server_model_params:
        grads.append((model_params[param_name].clone().detach().flatten() - server_model_params[param_name].clone().detach().flatten()) / lr) 
    return torch.cat(grads)

def set_grads_(model, server_model, new_grads, lr):
    start = 0
    server_model_params = server_model.model.state_dict()
    model_params = model.model.state_dict()
    for param_name in server_model_params:
        dims = model_params[param_name].shape
        end = start + dims.numel()
        model_params[param_name].copy_(server_model_params[param_name].clone().detach() + new_grads[start:end].reshape(dims).clone() * lr)  # w1' = w0 + (-g')
        start = end
    model.model.load_state_dict(model_params)
    return model

def pcgrad(client_grads):
    pc_grad = copy.deepcopy(client_grads)
    for g_i in pc_grad:
        random.shuffle(client_grads)
        for g_j in client_grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= g_i_g_j * g_j / (g_j.norm() ** 2)
    return pc_grad

def gradient_surgery(model, models, lr):
    grads = []
    for client_model in models:
        grads.append(get_grads_(client_model, model, lr))
    new_grads = pcgrad(grads)
    output_models = []
    for i, client_model in enumerate(models):
        output_model = set_grads_(client_model, model, new_grads[i], lr)
        output_models.append(output_model)
    return output_models

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
        
        models = gradient_surgery(model, models, lr)
        ensemble_model_params = average_models(models)
        model.model.load_state_dict(ensemble_model_params)
        
        name = 'FedGH_numlocalstep=' + str(num_local_step) + "_numclient=" + str(num_client) + '_numcomm=' + str(_) + '.pt'
        torch.save(model, name)

    torch.save(model,'FedGH_final_model.pt')
