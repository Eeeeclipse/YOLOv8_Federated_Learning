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

def evaluate_map50(trainedmodel, data_path, dataset='val'):
    metrics = trainedmodel.val(data=data_path, split=dataset, workers = 0)
    map50 = round(metrics.box.map50, 3)
    print("The mAP of model on {0} dataset is {1}".format(dataset,map50))
    return metrics, map50

curr_path=os.getcwd()
config_path = os.path.join(curr_path, 'config.yaml')


if __name__ == '__main__':
    FedAvg_map50_list = []
    FedGH_map50_list = []
    for i in range(31):
        if (i == 0):
            FedGH_model = YOLO('./yolov8n.yaml').load('yolov8n.pt')
            FedAvg_model = YOLO('./yolov8n.yaml').load('yolov8n.pt')
        else:
            FedGH_model_name = './FedGH_numlocalstep=1_numclient=4_numcomm=' + str(i - 1) + '.pt'
            FedAvg_model_name = './FedAvg_numlocalstep=1_numclient=4_numcomm=' + str(i - 1) + '.pt'
            FedGH_model = torch.load(FedGH_model_name)
            FedAvg_model = torch.load(FedAvg_model_name)
        FedGH_metrics, FedGH_map50 = evaluate_map50(FedGH_model, config_path, dataset='test')
        FedAvg_metrics, FedAvg_map50 = evaluate_map50(FedAvg_model, config_path, dataset='test')
        FedAvg_map50_list.append(FedAvg_map50)
        FedGH_map50_list.append(FedGH_map50)
    plt.figure(figsize=(10,8))
    plt.plot(FedAvg_map50_list, label = 'FedAvg')
    plt.plot(FedGH_map50_list, label = 'FedGH')
    plt.legend()

    plt.title('numlocalstep=1_numclient=4')
    plt.xlabel("communication round")
    plt.ylabel("mAP50")
    plt.show()
