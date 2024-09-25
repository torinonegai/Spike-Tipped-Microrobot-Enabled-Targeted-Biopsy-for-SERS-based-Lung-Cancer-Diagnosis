import torch
import numpy as np
from torch import nn
from RamanDataloader import RamanDataset
from sklearn.metrics import confusion_matrix

model = torch.load('model.pth', weights_only=False)
model.eval()
device = "cuda"

test_dataset = RamanDataset("data/test_data.txt","data/test_label.txt")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        predicted = pred.argmax(1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y.cpu().numpy())

conf_matrix_ = confusion_matrix(y_true, y_pred, normalize="true")
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(conf_matrix_,"\n")
print("Average accuracy: ", accuracy)