from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import numpy as np
import torch
import librosa

class wavDataset(Dataset):
    def __init__(self, dataset, args):
        self.labels = dataset['labels']
        self.data = dataset['input_values']
        self.args = args

    def __getitem__(self, index):
        
        tmp = self.data[index]
        tmp = librosa.util.normalize(tmp)
        if self.args.da:
            tmp = tmp + 0.009*np.random.normal(0,1,len(tmp))

        # zero_np = np.zeros(161000)
        # zero_np[:len(tmp)] = tmp
        zero_np = tmp
        
        tmp = torch.FloatTensor(zero_np)
        label = torch.tensor(int(self.labels[index]))

        return tmp, label

    def __len__(self):
        return len(self.labels)