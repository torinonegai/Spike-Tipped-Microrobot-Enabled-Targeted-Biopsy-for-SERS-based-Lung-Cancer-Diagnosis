import numpy as np
import torch
from torch.utils.data import Dataset
class RamanDataset(Dataset):
    def __init__(self,data_filename,label_filename,transform=None, target_transform=None):
        self.data = np.loadtxt(data_filename,dtype=np.float32)
        self.label = np.loadtxt(label_filename,dtype=np.int64)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return np.size(self.label,0)

    def __getitem__(self, idx):
        data = torch.unsqueeze(torch.from_numpy(self.data[idx,:]),0)
        label = torch.tensor(self.label[idx])
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data,label