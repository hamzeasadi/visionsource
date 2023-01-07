import numpy as np
from utils import KeepTrack
import conf as cfg
from Derrickpreprocessing import get_loader
# from camsrc0 import model
import camsrc1 as model
import engine
import argparse
import torch
from torch import nn as nn, optim as optim
from torch.utils.data import DataLoader
import os
from testing import final_result

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser(prog='train.py', description='required flags and supplemtary parameters for training')
parser.add_argument('--train', action=argparse.BooleanOptionalAction)
parser.add_argument('--test', action=argparse.BooleanOptionalAction)
# parser.add_argument('--val', action=argparse.BooleanOptionalAction)
parser.add_argument('--epoch', '-e', type=int, required=False, metavar='epoch', default=1)

args = parser.parse_args()


def train(net, train_loader, val_loader, opt, criterion, epochs, minerror, modelname:str):

    kt = KeepTrack(path=cfg.paths['model'])
    for epoch in range(epochs):
        train_loss = engine.train_step(model=net, data=train_loader, criterion=criterion, optimizer=opt)
        val_loss = engine.val_step(model=net, data=val_loader, criterion=criterion)
        if val_loss < minerror:
            minerror = val_loss
            kt.save_ckp(model=net, opt=opt, epoch=epoch, minerror=val_loss, fname=modelname)
            
        print(f"train_loss={train_loss} val_loss={minerror}")
    


def main():
    model_name = f"vision_Derrick_residual_2.pt"
    keeptrack = KeepTrack(path=cfg.paths['model'])
    Net = model.ConstConv(lcnf=cfg.constlayer)
    # Net = nn.DataParallel(Net)
    Net.to(dev)
    opt = optim.Adam(params=Net.parameters(), lr=3e-4)
    # criteria = OrthoLoss()
    criteria = nn.CrossEntropyLoss()
    train_loader, test_loader = get_loader(srcdatapath=cfg.paths['Derrickdata'])
    minerror = np.inf
    # if False:
    if args.train:
        train(net=Net, train_loader=train_loader, val_loader=test_loader, opt=opt, criterion=criteria, epochs=args.epoch, minerror=minerror, modelname=model_name)

    # if True:
    if args.test:
        # model_name = f"vision_residual_1.pt"
        state = keeptrack.load_ckp(fname=model_name)
        Net.load_state_dict(state['model'], strict=False)
        print(f"min error is {state['minerror']} which happen at epoch {state['epoch']}")
        engine.test_step(model=Net, data=test_loader, criterion=criteria)
        # final_result(model=Net, criterion=criteria, num_cls=9)



if __name__ == '__main__':
    main()