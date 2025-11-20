import torch
import torch.nn as nn
import torch.nn.functional as F


class GaitTCN(nn.Module):
    """
    Improved Temporal Convolutional Network for gait classification.
    Changes from original:
    1. Non-causal convolutions (can look forward and backward in the window)
    2. Larger fully connected layer for more capacity
    3. Better initialization
    4. Optional squeeze-and-excitation blocks for channel attention
    """
    
    def __init__(self, 
                 input_length=100,
                 input_channels=24,
                 num_classes=8,
                 num_channels=[64, 128, 256],
                 kernel_size=7,
                 dropout_rate=0.3,
                 fc_neurons=2048,  # Added larger FC layer
                 use_se_blocks=True,  # Channel attention
                 causal=False):  # Make non-causal by default
        """
        Args:
            input_length: Time-window length
            input_channels: Number of input features/sensors
            num_classes: Number of gait classes to predict
            num_channels: List of channel sizes for each TCN layer
            kernel_size: Convolutional kernel size (temporal dimension)
            dropout_rate: Dropout probability for regularization
            fc_neurons: Number of neurons in fully connected layer
            use_se_blocks: Whether to use squeeze-and-excitation blocks
            causal: If True, use causal convolutions (only look backward)
        """
        super().__init__()
        
        self.causal = causal
        
        # Input projection layer
        self.input_proj = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_channels[0],
            kernel_size=1
        )
        self.input_bn = nn.BatchNorm1d(num_channels[0])
        
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
                    dropout=dropout_rate,
                    use_se=use_se_blocks,
                    causal=causal
                )
            )
        
        # Enhanced classification head
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.adaptive_max_pool = nn.AdaptiveMaxPool1d(1)  # Add max pooling too
        
        # Larger fully connected layer for more capacity
        self.fc1 = nn.Linear(num_channels[-1] * 2, fc_neurons)  # *2 for avg+max pooling
        self.bn_fc = nn.BatchNorm1d(fc_neurons)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(fc_neurons, num_classes)
        
        # Store params for reference
        self.input_length = input_length
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
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
        x = F.relu(self.input_bn(self.input_proj(x)))
        
        # Pass through TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Dual pooling (average + max) for richer features
        x_avg = self.adaptive_pool(x)
        x_max = self.adaptive_max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Enhanced classification head
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Helps the network learn which channels are most important.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (batch, channels, time)
        batch, channels, time = x.size()
        
        # Squeeze: Global average pooling
        squeeze = F.adaptive_avg_pool1d(x, 1).view(batch, channels)
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation)).view(batch, channels, 1)
        
        # Scale the input
        return x * excitation


class TemporalBlock(nn.Module):
    """
    Improved TCN block with optional non-causal convolutions and SE blocks.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout, 
                 use_se=True, causal=False):
        super().__init__()
        
        self.causal = causal
        
        # For non-causal: padding on both sides
        # For causal: padding only on the left (past)
        if causal:
            padding = (kernel_size - 1) * dilation
        else:
            padding = ((kernel_size - 1) * dilation) // 2
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding if not causal else 0,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Only use Chomp if causal
        self.chomp1 = Chomp1d(padding) if causal else None
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding if not causal else 0,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.chomp2 = Chomp1d(padding) if causal else None
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation block
        self.se = SqueezeExcitation(out_channels) if use_se else None
        
        # Residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        
        self.relu_out = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch, channels, time)
        
        Returns:
            Output tensor (batch, channels, time)
        """
        # Causal: need to pad manually before conv
        if self.causal:
            padding = (self.conv1.kernel_size[0] - 1) * self.conv1.dilation[0]
            x_padded = F.pad(x, (padding, 0))
            out = self.conv1(x_padded)
        else:
            out = self.conv1(x)
        
        out = self.bn1(out)
        if self.chomp1 is not None:
            out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv
        if self.causal:
            padding = (self.conv2.kernel_size[0] - 1) * self.conv2.dilation[0]
            out_padded = F.pad(out, (padding, 0))
            out = self.conv2(out_padded)
        else:
            out = self.conv2(out)
        
        out = self.bn2(out)
        if self.chomp2 is not None:
            out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Apply SE block if enabled
        if self.se is not None:
            out = self.se(out)
        
        # Residual path
        res = x if self.downsample is None else self.downsample(x)
        
        # Add residual and apply final activation
        return self.relu_out(out + res)


class Chomp1d(nn.Module):
    """
    Removes padding from the end of the sequence to maintain causality.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x