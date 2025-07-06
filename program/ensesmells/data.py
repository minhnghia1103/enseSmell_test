# pytorch lib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ds lib
import numpy as np

# scikit-learn lib
from sklearn.utils import compute_class_weight

class Dataset(Dataset):
    def __init__(self, inputs, labels, is_test=False):
        self.is_test = is_test
        self.inputs = inputs
        self.labels = labels
    
    def __getitem__(self, idx):
        sample_input = torch.tensor(self.inputs[idx].reshape(1, -1), dtype=torch.float)
        if self.is_test:
            return sample_input
        else:
            sample_label = torch.tensor([self.labels[idx]], dtype=torch.float)
            return (sample_input, sample_label)
    
    def __len__(self):
        return self.inputs.shape[0]

class DatasetCombine(Dataset):
    def __init__(self, inputs, inputs_metrics, labels, is_test=False):
        self.is_test = is_test
        self.inputs = inputs
        self.inputs_metrics = inputs_metrics
        self.labels = labels
    
    def __getitem__(self, idx):
        sample_input = torch.tensor(self.inputs[idx].reshape(1, -1), dtype=torch.float)
        sample_input_metrics = torch.tensor(self.inputs_metrics[idx], dtype=torch.float)
        if self.is_test:
            return sample_input
        else:
            sample_label = torch.tensor([self.labels[idx]], dtype=torch.float)
            return (sample_input, sample_input_metrics, sample_label)
    
    def __len__(self):
        return self.inputs.shape[0]

class DatasetCombineWithGraph(Dataset):
    def __init__(self, inputs, inputs_metrics, labels, graph_data=None, is_test=False):
        self.is_test = is_test
        self.inputs = inputs
        self.inputs_metrics = inputs_metrics
        self.labels = labels
        self.graph_data = graph_data
    
    def __getitem__(self, idx):
        sample_input = torch.tensor(self.inputs[idx].reshape(1, -1), dtype=torch.float)
        sample_input_metrics = torch.tensor(self.inputs_metrics[idx], dtype=torch.float)
        
        if self.is_test:
            if self.graph_data is not None:
                return sample_input, sample_input_metrics, self.graph_data[idx]
            else:
                return sample_input, sample_input_metrics
        else:
            sample_label = torch.tensor([self.labels[idx]], dtype=torch.float)
            if self.graph_data is not None:
                return (sample_input, sample_input_metrics, sample_label, self.graph_data[idx])
            else:
                return (sample_input, sample_input_metrics, sample_label)
    
    def __len__(self):
        return self.inputs.shape[0]