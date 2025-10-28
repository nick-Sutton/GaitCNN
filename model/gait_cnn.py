import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitCNN(nn.Module):
    """
    CNN for gait/locomotion mode classification based on:
    "Integral Real-time Locomotion Mode Recognition Based on GA-CNN 
    for Lower Limb Exoskeleton" (Wang et al., 2022)
    """
    
    def __init__(self, 
                 input_length=100,      # Time-window length
                 input_channels=24,
                 num_classes=3,
                 # Conv1 params
                 f11=13, f12=7, w1=20,
                 # Pool1 params
                 p11=3, p12=3,
                 # Conv2 params
                 f21=11, f22=11, w2=15,
                 # Pool2 params
                 p21=3, p22=3,
                 # Fully connected params
                 fc_neurons=4576,
                 dropout_rate=0.5):
        
        super(GaitCNN, self).__init__()
        
        # Convolutional Layer 1
        # Each kernel/filter learns to detect different features
        # w1=20 means we learn 20 different feature detectors
        
        self.conv1 = nn.Conv2d(
            in_channels=1,           # Input: 1 channel
            out_channels=w1,         # Output: w1 feature maps
            kernel_size=(f11, f12),  # Each kernel is 14x7)
            stride=1,                # Slide 1 position at a time
            padding='same'           # Keep output same size as input
        )
        # Output shape: (batch, 20, 100, 24) - 20 different learned features
        
        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(
            kernel_size=(p11, p12),
            stride=2
        )
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=w1,
            out_channels=w2,
            kernel_size=(f21, f22),
            stride=1,
            padding='same'
        )
        
        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(
            kernel_size=(p21, p22),
            stride=2
        )

        # TODO: Condiser removing adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.flatten_size = w2 * 4 * 4
        
        # Fully Connected Layer
        self.fc1 = nn.Linear(self.flatten_size, fc_neurons)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Output Layer
        self.fc2 = nn.Linear(fc_neurons, num_classes)
        
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
        # Add channel dimension: (batch, 1, length, channels)
        x = x.unsqueeze(1)
        
        # Conv1 + ReLU
        x = F.relu(self.conv1(x))
        
        # Pool1
        x = self.pool1(x)
        
        # Conv2 + ReLU
        x = F.relu(self.conv2(x))
        
        # Pool2
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Output layer (logits for CrossEntropyLoss)
        x = self.fc2(x)
        
        return x