import numpy as np
import torch
import torch.nn as nn
from RamanDataloader import RamanDataset
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients

model = torch.load('tmp.pth').to(torch.device("cpu"))
model.eval()
train_dataset = RamanDataset("data/train_data.txt","data/train_label.txt")
total = train_dataset.__getitem__(range(0,2249))[0].squeeze()
normal = torch.reshape(torch.mean(total[0:447,:], 0),(1,1,-1))
ig = IntegratedGradients(model)
mean = torch.zeros(5,1106);
# Big cell carcinoma
attr = torch.zeros(432,1106);
for i in range(447,879):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=1, n_steps=50)
    attributions = attributions.squeeze()
    attr[i-447,:] = attributions
mean[1,:] = torch.mean(attr,0)
# Small cell carcinoma
attr = torch.zeros(527,1106);
for i in range(879,1406):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=2, n_steps=50)
    attributions = attributions.squeeze()
    attr[i-879,:] = attributions
mean[2,:] = torch.mean(attr,0)
# Squamous cell lung carcinoma
attr = torch.zeros(373,1106);
for i in range(1406,1779):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=3, n_steps=50)
    attributions = attributions.squeeze()
    attr[i-1406,:] = attributions
mean[3,:] = torch.mean(attr,0)
# Lung adeno-carcinoma
attr = torch.zeros(469,1106);
for i in range(1779,2248):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=4, n_steps=50)
    attributions = attributions.squeeze()
    attr[i-1779,:] = attributions
mean[4,:] = torch.mean(attr,0)

np.savetxt("attribution.txt",mean)
plt.plot(range(1106),mean.t())
plt.show()
print('IG Attributions:', attributions)