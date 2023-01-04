import torch
from torch import nn as nn
from torch import optim as optim
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
from torchmetrics import Accuracy
from sklearn.metrics import accuracy_score
from itertools import combinations


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

    Ypre = Y_pred.cpu().detach().numpy()
    Ytrue = Y_true.cpu().detach().numpy()
    for i in range(9):
        comps = combinations(iterable=Ypre[Ytrue==i], r=3)
        avgclsacc = 0
        j=0
        for comp in comps:
            j+=1
            avg = sum(np.array(comp)==i)
            if avg>1:
                avgclsacc += 1
        print(f"class={i} ---> accuracy={avgclsacc/j}")



def main():
    y = np.random.randint(low=0, high=3, size=(20,))
    ypre = np.random.randint(low=0, high=3, size=(20,))
    yy = []
    comps = []
    accs = []
    for i in range(3):
        comps = combinations(iterable=ypre[y==i], r=3)
        avgclsacc = 0
        j=0
        for comp in comps:
            j+=1
            avg = sum(np.array(comp)==i)
            if avg>1:
                avgclsacc += 1
        print(avgclsacc/j)


        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        
    
    

    
    
    
    



if __name__ == '__main__':
    main()