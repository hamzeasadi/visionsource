import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
from preprocessing import bgr2graycoord, imgpatchs
import os
import conf as cfg
import random
from torchvision import transforms
import cv2
from torchvision.datasets import ImageFolder


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    model.to(dev)
    Y_true = torch.tensor([1], device=dev)
    Y_pred = torch.tensor([1], device=dev)
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out = model(X)
            yhat = torch.argmax(out, dim=1)
            Y_true = torch.cat((Y_true, Y))
            Y_pred = torch.cat((Y_pred, yhat))

    print(Y_pred.shape, Y_true.shape)
    acc = accuracy_score(Y_pred.cpu().detach().numpy(), Y_true.cpu().detach().numpy())
    print(f"acc is {acc}")


# def extract_all_patches(srcpath, trgpath):
#     srcfolders = os.listdir(srcpath)

#     try:
#         srcfolders.remove('.DS_Store')
#     except Exception as e:
#         print(e)

#     for srcfolder in srcfolders:


def extractiframes(srcpath, trgpath):
    srcfolders = os.listdir(srcpath)

    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print(e)

    for srcfolder in srcfolders:
        trgfolderpath = os.path.join(trgpath, srcfolder)
        srcfolderpath = os.path.join(srcpath, srcfolder)
        srcfolderfiles = os.listdir(srcfolderpath)

        try:
            srcfolderfiles.remove('.DS_Store')
        except Exception as e:
            print(e)
        i=0
        for srcfile in srcfolderfiles:
            videofilepath = os.path.join(srcfolderpath, srcfile)
            trgfilepath = os.path.join(trgfolderpath, f'iamge-{i}-')
            os.system(f"ffmpeg -skip_frame nokey -i {videofilepath} -vsync 0 -frame_pts true {trgfilepath}%d.png")
            i+=1


def createdb(srcpath, trgpath):
    srcfolders = os.listdir(srcpath)

    try:
        srcfolders.remove('.DS_Store')
    except Exception as e:
        print(e)

    for srcfolder in srcfolders:
        srcfolderpath = os.path.join(srcpath, srcfolder)
        srcfolderfiles = os.listdir(srcfolderpath)
        randomsample = random.sample(srcfolderfiles, 146)

        for i, srcfile in enumerate(randomsample):
            trgfolderpath = os.path.join(trgpath, f'folder_{i}', srcfolder)
            cfg.creatdir(trgfolderpath)
            srcfilepath = os.path.join(srcfolderpath, srcfile)
            img = cv2.imread(srcfilepath)
            if img is not None:
                grayimag = bgr2graycoord(img)
                patches = imgpatchs(img=grayimag)

                for j, patch in enumerate(patches):
                    patchpath = os.path.join(trgfolderpath, f'patch_{j}.png')
                    cv2.imwrite(filename=patchpath, img=patch)


def createtestdata(folder_id):
    t = transforms.Compose([transforms.ToTensor()])
    foldername = f"folder_{folder_id}"
    folderpath = os.path.join(cfg.paths['datatest'], foldername)
    dataset = ImageFolder(root=folderpath, transform=t)
    return  DataLoader(dataset=dataset, batch_size=500)




def main():
    y = torch.tensor([1])

    # for i in range(10):
    #     x = torch.randn(size=(5, 6))
    #     yhat = torch.argmax(x, dim=1)
    #     y = torch.cat((y, yhat))

   
    srcpath = cfg.paths['videotestiframes']
    trgpath = cfg.paths['datatest']

    # createdb(srcpath=srcpath, trgpath=trgpath)

    firstfolder = createtestdata(folder_id=1)
    batch = next(iter(firstfolder))
    print(batch[0].shape, batch[1])


    



if __name__ == '__main__':
    main()