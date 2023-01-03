import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from torchmetrics import Accuracy


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model: nn.Module, data: DataLoader, criterion: nn.Module, optimizer: optim):
    epoch_error = 0
    l = len(data)
    model.train()
    for i, (X, Y) in enumerate(data):
        X = X.to(dev)
        Y = Y.to(dev)
        out = model(X)
        loss = criterion(out, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_error += loss.item()
        # break
    return epoch_error/l


def val_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out = model(X)
            loss = criterion(out, Y)
            epoch_error += loss.item()
            # break
    return epoch_error/l


def test_step(model: nn.Module, data: DataLoader, criterion: nn.Module):
    epoch_error = 0
    l = len(data)
    model.eval()
    model.to(dev)
    Y_true = torch.tensor([1])
    Y_pred = torch.tensor([1])
    with torch.no_grad():
        for i, (X, Y) in enumerate(data):
            X = X.to(dev)
            Y = Y.to(dev)
            out = model(X)
            yhat = torch.argmax(out, dim=1)
            Y_true = torch.cat((Y_true, Y))
            Y_pred = torch.cat((Y_pred, yhat))
            # loss = criterion(out, Y)
            # epoch_error += loss.item()


    accuracy = Accuracy(task='multiclass', num_classes=6)
    # y = Y.numpy()
    print(Y_pred.shape, Y_true.shape)
    # print(Y)
    # print(out)
    
    
    acc = accuracy(Y_pred.detach(), Y_true.detach())
    print(f"acc is {acc}")

    # yhat = torch.argmax(out, dim=1)
    # print(Y)
    # print(yhat)

def main():
    y = torch.tensor([1])
    for i in range(10):
        x = torch.randn(size=(5, 6))
        yhat = torch.argmax(x, dim=1)
        y = torch.cat((y, yhat))

    



if __name__ == '__main__':
    main()