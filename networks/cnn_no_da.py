import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
torch.manual_seed(222222)
np.random.seed(222222)
# This was the architecture I got the best results on
class NeuralNetworkNoDA(nn.Module):
    def __init__(self):
        super(NeuralNetworkNoDA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=(1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16*2, kernel_size=3, stride=(1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(16*2)
        self.conv3 = nn.Conv2d(in_channels=16*2, out_channels=16*4, kernel_size=3, stride=(1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(16*4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(in_features=16*4*8*8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)
        
    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        
        x = x.view(-1, 16*4*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        x = x.view(-1, 4)
        return x