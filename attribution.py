import numpy as np
import torch
import torch.nn as nn
from RamanDataloader import RamanDataset
import matplotlib.pyplot as plt

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

model = torch.load('tmp.pth').to(torch.device("cpu"))
model.eval()
train_dataset = RamanDataset("data/train_data.txt","data/train_label.txt")
total = train_dataset.__getitem__(range(0,2249))[0].squeeze()
normal = torch.reshape(torch.mean(total[0:447,:], 0),(1,1,-1))
ig = IntegratedGradients(model)
mean = torch.zeros(5,1106);

fck = torch.zeros(432,1106);
for i in range(447,879):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=1, n_steps=50)
    attributions = attributions.squeeze()
    fck[i-447,:] = attributions
mean[1,:] = torch.mean(fck,0)

fck = torch.zeros(527,1106);
for i in range(879,1406):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=2, n_steps=50)
    attributions = attributions.squeeze()
    fck[i-879,:] = attributions
mean[2,:] = torch.mean(fck,0)

fck = torch.zeros(373,1106);
for i in range(1406,1779):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=3, n_steps=50)
    attributions = attributions.squeeze()
    fck[i-1406,:] = attributions
mean[3,:] = torch.mean(fck,0)

fck = torch.zeros(469,1106);
for i in range(1779,2248):
    data = torch.reshape(total[i,:],(1,1,-1))
    attributions = ig.attribute(data, normal, target=4, n_steps=50)
    attributions = attributions.squeeze()
    fck[i-1779,:] = attributions
mean[4,:] = torch.mean(fck,0)

np.savetxt("fuck.txt",mean)
plt.plot(range(1106),mean.t())
plt.show()
print('IG Attributions:', attributions)
# print('Convergence Delta:', delta)
