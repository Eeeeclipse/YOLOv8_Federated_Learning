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
import torch
import cv2
import torch
from pathlib import Path

def_size = 640
def get_bbox_from_label(text_file_path):
    bbox_list=[]
    with open(text_file_path, "r") as file:
        for line in file:
            _,x_centre,y_centre,width,height=line.strip().split(" ")
            x1=(float(x_centre)+(float(width)/2))*def_size
            x0=(float(x_centre)-(float(width)/2))*def_size
            y1=(float(y_centre)+(float(height)/2))*def_size
            y0=(float(y_centre)-(float(height)/2))*def_size
            vertices=np.array([[int(x0), int(y0)], [int(x1), int(y0)], 
                               [int(x1),int(y1)], [int(x0),int(y1)]])
            bbox_list.append(vertices)          
    return tuple(bbox_list)


if __name__ == '__main__':
    FedGH_model4_name = './FedGH_numlocalstep=1_numclient=4_numcomm=' + str(4) + '.pt'
    FedAvg_model4_name = './FedAvg_numlocalstep=1_numclient=4_numcomm=' + str(4) + '.pt'
    FedGH_model4 = torch.load(FedGH_model4_name)
    FedAvg_model4 = torch.load(FedAvg_model4_name)

    FedGH_model9_name = './FedGH_numlocalstep=1_numclient=4_numcomm=' + str(9) + '.pt'
    FedAvg_model9_name = './FedAvg_numlocalstep=1_numclient=4_numcomm=' + str(9) + '.pt'
    FedGH_model9 = torch.load(FedGH_model9_name)
    FedAvg_model9 = torch.load(FedAvg_model9_name)

    FedGH_model29_name = './FedGH_numlocalstep=1_numclient=4_numcomm=' + str(29) + '.pt'
    FedAvg_model29_name = './FedAvg_numlocalstep=1_numclient=4_numcomm=' + str(29) + '.pt'
    FedGH_model29 = torch.load(FedGH_model29_name)
    FedAvg_model29 = torch.load(FedAvg_model29_name)

    # simple task 00002848
    # difficult task 00001890
    image1 = cv2.imread('./images/validation/00001890.jpg')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1_copy = copy.deepcopy(image1)
    bbox1 = get_bbox_from_label('./labels/validation/00001890.txt')
    FedGH_image1_4 = FedGH_model4(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    FedGH_image1_9 = FedGH_model9(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    FedGH_image1_29 = FedGH_model29(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    FedAvg_image1_4 = FedAvg_model4(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    FedAvg_image1_9 = FedAvg_model9(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    FedAvg_image1_29 = FedAvg_model29(image1, augment=False, visualize=False)[0].plot('Arial.ttf')
    red=(255,0,0) 
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 4, 1)
    cv2.drawContours(image1_copy, bbox1, -1, red, 2)
    plt.imshow(image1_copy)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Sample output')

    plt.subplot(2, 4, 2)
    plt.imshow(FedAvg_image1_4)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedAvg_round=5')

    plt.subplot(2, 4, 3)
    plt.imshow(FedAvg_image1_9)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedAvg_round=10')

    plt.subplot(2, 4, 4)
    plt.imshow(FedAvg_image1_29)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedAvg_round=30')

    plt.subplot(2, 4, 6)
    plt.imshow(FedGH_image1_4)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedGH_round=5')

    plt.subplot(2, 4, 7)
    plt.imshow(FedGH_image1_9)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedGH_round=10')

    plt.subplot(2, 4, 8)
    plt.imshow(FedGH_image1_29)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('FedGH_round=30')

    plt.show()