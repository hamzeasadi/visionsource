import torch
from torch import nn as nn
from torch.nn import functional as F
from torchinfo import summary
from torchvision import models

import os, sys
sys.path.append(os.pardir)
import conf as cfg


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelBase(nn.Module):
    def __init__(self, name, created_time):
        super(ModelBase, self).__init__()
        self.name = name
        self.created_time = created_time

    def copy_params(self, state_dict):
        own_state = self.state_dict()
        for (name, param) in state_dict.items():
            if name in own_state:
                own_state[name].copy_(param.clone())

    def boost_params(self, scale=1.0):
        if scale == 1.0:
            return self.state_dict()
        for (name, param) in self.state_dict().items():
            self.state_dict()[name].copy_((scale * param).clone())
        return self.state_dict()

    # self - x
    def sub_params(self, x):
        own_state = self.state_dict()
        for (name, param) in x.items():
            if name in own_state:
                own_state[name].copy_(own_state[name] - param)

    # self + x
    def add_params(self, x):
        a = self.state_dict()
        for (name, param) in x.items():
            if name in a:
                a[name].copy_(a[name] + param)


class ConstConv(ModelBase):
    """
    doc
    """
    def __init__(self, lcnf: dict, name='constlayer', created_time=None):
        super().__init__(name=name, created_time=created_time)
        self.lcnf = lcnf
        self.register_parameter("const_weight", None)
        self.const_weight = nn.Parameter(torch.randn(size=[lcnf['outch'], 1, lcnf['ks'], lcnf['ks']]), requires_grad=True)
        self.coord = self.coords(h=480, w=800)
        self.fx = self.feat_ext()

        # resnet_weight = models.ResNet50_Weights.DEFAULT
        # self.base_model = models.resnet50(weights=resnet_weight)
        # self.base_model.fc = nn.Linear(in_features=2048, out_features=9)
        # self.const2res = nn.Conv2d(in_channels=lcnf['outch']+2, out_channels=3, kernel_size=3, stride=1, padding='same')

    def add_pos(self, res):
        Z = []
        for i in range(res.shape[0]):
            residual = res[i, :, :, :]
            z = torch.cat((residual, self.coord), dim=0)
            Z.append(z.unsqueeze_(dim=0)) 
        return torch.cat(tensors=Z, dim=0)

    def coords(self, h, w):

        channelx = torch.randn(size=(h, w), device=dev)
        for i in range(h):
            channelx[i, :] = i*channelx[i, :]
        channelx = 2*(channelx/h) - 1

        channely = torch.randn(size=(h, w), device=dev)
        for i in range(w):
            channely[:, i] = i*channely[:, i]
        channely = 2*(channely/w) - 1

        return torch.cat((channelx.unsqueeze_(dim=0), channely.unsqueeze_(dim=0)), dim=0)

        


    def normalize(self):
        cntrpxl = int(self.lcnf['ks']/2)
        centeral_pixels = (self.const_weight[:, 0, cntrpxl, cntrpxl])
        for i in range(self.lcnf['outch']):
            sumed = (self.const_weight.data[i].sum() - centeral_pixels[i])/self.lcnf['scale']
            self.const_weight.data[i] /= sumed
            self.const_weight.data[i, 0, cntrpxl, cntrpxl] = -self.lcnf['scale']

    def feat_ext(self):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=self.lcnf['outch']+2, out_channels=96, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(num_features=96),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=5, stride=1, padding='same'), nn.BatchNorm2d(num_features=64),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding='same'), nn.BatchNorm2d(num_features=64),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=128),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding='same'), nn.BatchNorm2d(num_features=128),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding='same'), nn.BatchNorm2d(num_features=256),
            nn.Tanh(), nn.MaxPool2d(kernel_size=3, stride=2),
           
            nn.Flatten(),
            nn.Linear(in_features=2560, out_features=1024), nn.Tanh(),
            # nn.Linear(in_features=1024, out_features=1024), nn.Tanh(),
            nn.Linear(in_features=1024, out_features=9)
        )

        return layer

    def forward(self, x):
        self.normalize()
        noise = F.conv2d(x[:, 0:1, :, :], self.const_weight, padding='same')
        noisecoord = self.add_pos(res=noise)
        x = self.fx(noisecoord)

        # x = self.const2res(noisecoord)
        # x = self.base_model(x)
        
        return x 


def cnnout(ks, stride, w):
    outsize = int((w-ks)/stride) + 1
    return outsize


def main():
    x = torch.randn(size=[2, 3, 480, 800])
    model = ConstConv(lcnf=cfg.constlayer)
    out = model(x)
    print(out.shape)
    summary(model, input_size=[10, 1, 480, 800])

  
    # cout1 = cnnout(5, 1, 224)
    # mout1 = maxout(ks=3, stride=2, w=cout1)

    # cout2 = cnnout(5, 1, mout1)
    # mout2 = maxout(ks=3, stride=2, w=cout2)

    # cout3 = cnnout(3, 1, mout2)
    # mout3 = maxout(ks=3, stride=2, w=cout3)

    # cout4 = cnnout(3, 1, mout3)
    # mout4 = maxout(ks=3, stride=2, w=cout4)

    # cout5 = cnnout(3, 1, mout4)
    # mout5 = maxout(ks=3, stride=2, w=cout5)

    # print(f'cout1={cout1}, mout1={mout1}')
    # print(f'cout2={cout2}, mout2={mout2}')
    # print(f'cout3={cout3}, mout3={mout3}')
    # print(f'cout4={cout4}, mout4={mout4}')
    # print(f'cout5={cout5}, mout5={mout5}')

if __name__ == '__main__':
    main()