import time
import torch
from torch import nn
from RamanDataloader import RamanDataset
from network import NeuralNetwork
from train_model import train_loop, test_loop
from plot_trainingprogress import update_training_plot, finish_plot
from torch.optim.lr_scheduler import ExponentialLR

""" Load Data """
train_dataset = RamanDataset("data/train_data.txt","data/train_label.txt")
test_dataset = RamanDataset("data/test_data.txt","data/test_label.txt")
'''
train_size = int(raman_dataset.__len__()*0.7)
test_size = raman_dataset.__len__()-train_size
train_dataset, test_dataset = torch.utils.data.random_split(raman_dataset,[train_size,test_size])
'''
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


""" Parameters """
device = "cuda"
model = NeuralNetwork().to(device)
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adma(model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = ExponentialLR(optimizer, gamma=0.95)
epochs = 200

""" Training """
train_loss, train_accuracy = test_loop(train_dataloader, model, loss_fn, device)
test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, device)
update_training_plot(0,train_loss, train_accuracy,test_loss, test_accuracy,50)

for t in range(1, epochs+1):
    # print(f"Epoch {t+1}\n-------------------------------")
    train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer, device)
    test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, device)
    update_training_plot(t,train_loss, train_accuracy,test_loss, test_accuracy,50)
    scheduler.step()
test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn, device)
#print("train:", train_loss, train_accuracy)
print("test:", test_loss, test_accuracy)

finish_plot();

torch.save(model, 'tmp.pth')