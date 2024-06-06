import torch
import torch.nn as nn
import numpy as np

class fNIRSInputLayer(nn.Module):
    def __init__(self, num_channels):
        super(fNIRSInputLayer, self).__init__()
        self.linear = nn.Linear(num_channels, 128) # Adjust output dimension as needed
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

# Example usage:
num_fNIRS_channels = 64 # Replace with your dataset's channel count
input_layer = fNIRSInputLayer(num_fNIRS_channels)
fNIRS_data = torch.randn(10, 64, 100) # Example fNIRS data shape
processed_data = input_layer(fNIRS_data)
