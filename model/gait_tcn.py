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


class TemporalBlock(nn.Module):
    """
    Single TCN block with dilated causal convolutions and residual connection.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        
        # Calculate padding for causal convolution
        padding = (kernel_size - 1) * dilation
        
        # First convolutional layer
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        
        # Second convolutional layer
        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 conv for residual connection if dimensions don't match
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.utils.weight_norm(
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            )
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor (batch, channels, time)
        
        Returns:
            Output tensor (batch, channels, time)
        """
        # Store residual
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Truncate to match input length (causal)
        out = out[:, :, :x.size(2)]
        
        # Apply residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = residual[:, :, :x.size(2)]
        
        out = out + residual
        out = F.relu(out)
        
        return out