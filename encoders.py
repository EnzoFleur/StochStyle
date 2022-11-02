import torch
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased")

class DAN(torch.nn.Module):

    def __init__(self, input_dim, hidden, r):
        super(DAN, self).__init__()

        self.input_dim = input_dim
        self.hidden = hidden
        self.r = r

        self.do1 = torch.nn.Dropout(0.1)
        self.bn1 = torch.nn.BatchNorm1d(input_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.do2 = torch.nn.Dropout(0.1)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.fc2 = torch.nn.Linear(hidden, r)

    def forward(self, x):

        x = x.mean(dim=1)
        x = self.do1(x)
        x = self.bn1(x)
        x = self.fc1(x)
        x = self.do2(x)
        x = self.bn2(x)
        x = self.fc2(x)

        return x


class MLP(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(MLP, self).__init__()

            self.bn = torch.nn.BatchNorm1d(input_size)
            self.do = torch.nn.Dropout(p=0.1)
            self.fc1 = torch.nn.Linear(input_size, output_size)
        
        def forward(self, x):
            x = x.mean(dim=1)
            x = self.bn(x)
            x = self.fc1(self.do(x))
            return x