import numpy as np
import torch
import torch.nn as nn
from RamanDataloader import RamanDataset
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

model = torch.load('tmp.pth').to(torch.device("cpu"))
model.eval()
train_dataset = RamanDataset("data/train_data_demo.txt","data/train_label_demo.txt")
total = train_dataset.__getitem__(range(500))[0].squeeze()
normal = torch.reshape(torch.mean(total[0:447,:], 0),(1,1,-1))
ig = IntegratedGradients(model)
mean = torch.zeros(5,1106);
attr = torch.zeros(500,1106);
for i in range(500):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=1, n_steps=50)
    attributions = attributions.squeeze()
    attr[i,:] = attributions
mean[1,:] = torch.mean(attr,0)

plt.plot(range(1106),mean.t())
plt.show()
