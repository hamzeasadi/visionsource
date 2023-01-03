import os
import conf as cfg
from torchvision.datasets import ImageFolder
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split


t = transforms.Compose([transforms.ToTensor()])

# traindataset = ImageFolder(root=cfg.paths['train'], transform=t)
# testdataset = ImageFolder(root=cfg.paths['test'], transform=t)

visionset = ImageFolder(root=cfg.paths['dataset'], transform=t)

def createdb(dataset: Dataset, batch_size=256, train_percent=0.80):
    l = len(dataset)
    train_size = int(l*train_percent)
    validation_size = l - train_size
    train, validation = random_split(dataset=dataset, lengths=[train_size, validation_size])

    val_size = int(0.8*validation_size)
    test_size = validation_size - val_size
    val, test = random_split(dataset=validation, lengths=[val_size, test_size])

    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

# trainl = DataLoader(dataset=traindataset, batch_size=128, shuffle=True)
# vall, testl = createdb(dataset=testdataset, batch_size=128)

trainloader, valloader, testloader = createdb(dataset=visionset, batch_size=128)

def main():
    # extract_patches(srcpath=cfg.paths['src_data'], trgpath=cfg.paths['images'])
    for X, Y in testloader:
        print(X.shape)

if __name__ == '__main__':
    main()