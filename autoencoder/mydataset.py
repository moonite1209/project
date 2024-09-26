import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        semantics_path=os.path.join(data_dir, 'raw_semantics.npy')
        self.data: torch.Tensor = torch.from_numpy(np.load(semantics_path)).squeeze(1)
        self.data = self.data.to(torch.float32)

    def __getitem__(self, index):
        return self.data[index].clone().detach()

    def __len__(self):
        return self.data.shape[0] 