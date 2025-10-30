import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated causal convolutions and residual connection.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First dilated causal conv
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)  # Remove future information
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal conv
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # 1x1 conv for residual connection if dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Removes padding from the end to ensure causality"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class GaitTCN(nn.Module):
    """
    TCN for gait/locomotion mode classification
    """
    
    def __init__(self, 
                 input_length=100,
                 input_channels=26,
                 num_classes=3,
                 num_channels=[64, 64, 128, 128],  # Hidden channels per layer
                 kernel_size=7,
                 dropout_rate=0.5):
        super(GaitTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        # Build TCN layers with increasing dilation
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential dilation: 1, 2, 4, 8...
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding to maintain sequence length
            padding = (kernel_size - 1) * dilation
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation, padding=padding,
                dropout=dropout_rate
            )]
        
        self.network = nn.Sequential(*layers)
        
        # Global average pooling over time dimension
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        # Store params for reference
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_classes = num_classes
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, time_length, num_channels)
               e.g., (batch, 100, 24)
               Data should be normalized to [0, 1]
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Transpose to (batch, channels, time) for Conv1d
        x = x.transpose(1, 2)  # (batch, 24, 100)
        
        # Apply TCN layers
        x = self.network(x)  # (batch, num_channels[-1], time)
        
        # Global average pooling
        x = self.gap(x)  # (batch, num_channels[-1], 1)
        
        # Flatten
        x = x.squeeze(-1)  # (batch, num_channels[-1])
        
        # Classification
        x = self.fc(x)  # (batch, num_classes)
        
        return x