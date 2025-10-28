import torch
from torch.utils.data import Dataset, DataLoader

class GaitDataset(Dataset):
    """PyTorch Dataset for gait data"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of shape (n_samples, window_length, channels)
            labels: numpy array of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
