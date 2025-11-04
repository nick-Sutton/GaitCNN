import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitTCN(nn.Module):
    """
    Temporal Convolutional Network for gait classification.
    Uses dilated causal convolutions with residual connections.
    """
    
    def __init__(self, 
                 input_length=100,
                 input_channels=24,
                 num_classes=3,
                 num_channels=[64, 128, 256],
                 kernel_size=7,
                 dropout_rate=0.3):
        """
        Args:
            input_length: Time-window length
            input_channels: Number of input features/sensors
            num_classes: Number of gait classes to predict
            num_channels: List of channel sizes for each TCN layer
            kernel_size: Convolutional kernel size (temporal dimension)
            dropout_rate: Dropout probability for regularization
        """
        super(GaitTCN, self).__init__()
        
        # Input projection layer
        self.input_proj = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_channels[0],
            kernel_size=1
        )
        
        # Build TCN blocks
        self.tcn_blocks = nn.ModuleList()
        for i, out_channels in enumerate(num_channels):
            in_ch = num_channels[i-1] if i > 0 else num_channels[0]
            dilation = 2 ** i
            
            self.tcn_blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout_rate
                )
            )
        
        # Adaptive pooling and classification head
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
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
        # Transpose: (batch, time, channels) -> (batch, channels, time)
        x = x.transpose(1, 2)
        
        # Input projection
        x = F.relu(self.input_proj(x))
        
        # Pass through TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dropout + Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class Chomp1d(nn.Module):
    """
    Removes padding from the end of the sequence to maintain causality.
    This ensures the convolution only looks at past timesteps, not future ones.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


class TemporalBlock(nn.Module):
    """
    Single TCN block with dilated causal convolutions and residual connection.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        # REMOVE weight_norm wrapper
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)  # ADD THIS
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)  # ADD THIS
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Order matters: Conv → Chomp → BatchNorm → ReLU → Dropout
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)  # ADD THIS
            )
    
    def init_weights(self):
        """
        Initialize weights with small values to prevent exploding gradients.
        Usi                nng normal distribution with std=0.01 for stable initialization.
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch, channels, time)
        
        Returns:
            Output tensor (batch, channels, time)
        """
        # Main path through convolutions
        out = self.net(x)
        
        # Residual path
        res = x if self.downsample is None else self.downsample(x)
        
        return out + res