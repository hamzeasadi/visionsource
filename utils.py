import os, sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import time
from torch import nn as nn
from torch import optim




dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KeepTrack():
    def __init__(self, path) -> None:
        self.path = path
        self.state = dict(model="", opt="", epoch=1, minerror=0.1)

    def save_ckp(self, model: nn.Module, opt: optim.Optimizer, epoch, minerror, fname: str):
        self.state['model'] = model.state_dict()
        self.state['opt'] = opt.state_dict()
        self.state['epoch'] = epoch
        self.state['minerror'] = minerror
        save_path = os.path.join(self.path, fname)
        torch.save(obj=self.state, f=save_path)

    def load_ckp(self, fname):
        state = torch.load(os.path.join(self.path, fname), map_location=dev)
        return state



def main():
    batch_size = 5

    
  

if __name__ == '__main__':
    main()








































# ab = np.array([[0, 0], [1, -1]])
# abmidepoint = np.mean(ab, axis=0)

# def sample_circle(center=abmidepoint, r: float=0.1, num_points=5):
#     thetas = np.linspace(start=0, stop=np.pi*2, num=num_points)
#     points = np.zeros(shape=(num_points, 2))
#     for i in range(num_points):
#         x = [r*np.cos(thetas[i]), r*np.sin(thetas[i])]
#         points[i] = center + x
    
#     return points
    

# def m_ab(knwonpoints, orthocenter):
#     # print(knwonpoints)
#     x1, y1, x2, y2 = knwonpoints[0, 0], knwonpoints[0, 1], knwonpoints[1, 0], knwonpoints[1, 1]
#     h1, h2 = orthocenter[0], orthocenter[1]
#     m = (y2 - y1)/(x2 - x1)
#     b = h2 + (1/m)*h1

#     x = (m*(b - y1)*(y2-h2) + m*x1*(h1-x2))/(m*(h1-x2) + (y2-h2))
#     y = (-1/m)*x + b

#     return x, y


# # orthos = sample_circle()
# # x, y = m_ab(knwonpoints=ab, orthocenter=orthos[0])


# def third_points(knownpoints, midepoint, r, num_points):
#     orthopoints = sample_circle(center=midepoint, r=r, num_points=num_points)
#     npoints = np.zeros_like(orthopoints)
#     for i, point in enumerate(orthopoints):
#         x = m_ab(knwonpoints=knownpoints, orthocenter=point)
#         npoints[i] = x

#     return npoints, orthopoints





# R = np.linspace(start=0.02, stop=0.1, num=5)
# Y = ['cyan', 'orange', 'red', 'yellow', 'black']

# plt.figure()
# plt.scatter(ab[:, 0], ab[:, 1], s=10, c='blue')
# plt.scatter(abmidepoint[0], abmidepoint[1], s=10, c='green')
# for j, r in enumerate(R):
#     zpoint, orthocenter = third_points(knownpoints=ab, midepoint=abmidepoint, r=r, num_points=10)
#     plt.scatter(orthocenter[:, 0], orthocenter[:, 1], s=10, c=Y[j], label=f"raduis={r}")
#     plt.scatter(zpoint[:, 0], zpoint[:, 1], s=10, c=Y[j])
# # t1 = plt.Polygon(X[:3,:], color=Y[0], fill=False)
# # plt.gca().add_patch(t1)

# # t2 = plt.Polygon(X[3:6,:], color=Y[3])
# # plt.gca().add_patch(t2)
# plt.legend()
# plt.show()



#     # plt.scatter(ab[:, 0], ab[:, 1], s=10, c='blue')
#     # plt.scatter(abmidepoint[0], abmidepoint[1], s=10, c='green')

#     # plt.scatter(orthocenter[:, 0], orthocenter[:, 1], s=10, c='cyan')
#     # plt.scatter(zpoint[:, 0], zpoint[:, 1], s=10, c='cyan')