import os, sys, random
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np


# seed intialization
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

root = os.getcwd()
datapath = os.path.join(root, 'data')
paths = dict(
    root=root, data=datapath, dataset=os.path.join(datapath, 'dataset'), videos=os.path.join(datapath, 'videos'), 
    iframes=os.path.join(datapath, 'iframes'), model=os.path.join(datapath, 'model'), videotest=os.path.join(datapath, 'videotest'),
    videotestiframes=os.path.join(datapath, 'videotestiframes'), videostest=os.path.join(datapath, 'videostest'), 
    datatest=os.path.join(datapath, 'datatest')
)

constlayer = dict(ks=5, scale=1, outch=8)

def creatdir(path: str):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f"{path} is already exist!!!!")




def main():
    pass


if __name__ == '__main__':
    main()
