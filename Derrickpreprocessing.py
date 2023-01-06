import os, sys, random
import cv2
import torch
from matplotlib import pyplot as plt
import conf as cfg
from torchvision import transforms
import numpy as np
import subprocess
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder


def crop_img(imgpath, H, W):
    img = cv2.imread(imgpath)
    h, w, c = img.shape
    centrh = int(h/2)
    centrw = int(w/2)
    dh = int(H/2)
    dw = int(W/2)
    return img[centrh-dh:centrh+dh, centrw-dw:centrw+dw, :]

def extract_data(srcpath, trgpath):
    srcfolders = os.listdir(srcpath)
    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print("already removed")

    for srcfolder in srcfolders:
        srcfolderpath = os.path.join(srcpath, srcfolder)
        trgfolderpath = os.path.join(trgpath, srcfolder)
        cfg.creatdir(trgfolderpath)
        srcfolderimages = os.listdir(srcfolderpath)
        for srcimg in srcfolderimages:
            srcimgpath = os.path.join(srcfolderpath, srcimg)
            trgimgpath = os.path.join(trgfolderpath, srcimg)
            crop = crop_img(srcimgpath, H=480, W=800)
            cv2.imwrite(filename=trgimgpath, img=crop)


def get_loader(srcdatapath):
    t = transforms.Compose(transforms=[transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=[100/255], std=[200/255])])
    data = ImageFolder(root=srcdatapath, transform=t)
    trainsize = int(len(data)*0.85)
    testsize = len(data) - trainsize
    train_data, test_data = random_split(dataset=data, lengths=[trainsize, testsize])
    trainl = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
    testl = DataLoader(dataset=test_data, batch_size=128)
    return trainl, testl



def main():
    print(2)
    src_path = cfg.paths['iframes']
    trg_path = cfg.paths['Derrickdata']
    # extract_data(srcpath=src_path, trgpath=trg_path)
    trainloader, testloader = get_loader(trg_path)
    for batch in testloader:
        print(batch[0].shape, batch[1])
        break


if __name__ == '__main__':
    main()
