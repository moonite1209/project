import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        semantics_path=os.path.join(data_dir, 'semantics.pt')
        self.data: torch.Tensor = torch.load(semantics_path, weights_only=True).squeeze(1)
        self.data = self.data.to(torch.float32)

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 