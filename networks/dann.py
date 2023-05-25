import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
torch.manual_seed(222222)
np.random.seed(222222)
# Gradient reversal layer for DANN architecure
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x):

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output
# This was the DANN architecture that got the best results
bf = 8
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=bf, kernel_size=3, stride=(1,1), padding='same')
        self.bn1 = nn.BatchNorm2d(bf)
        self.conv2 = nn.Conv2d(in_channels=bf, out_channels=bf*2, kernel_size=3, stride=(1,1), padding='same')
        self.bn2 = nn.BatchNorm2d(bf*2)
        self.conv3 = nn.Conv2d(in_channels=bf*2, out_channels=bf*4, kernel_size=3, stride=(1,1), padding='same')
        self.bn3 = nn.BatchNorm2d(bf*4)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.)
        
        # Task
        self.tnfc1 = nn.Linear(in_features=bf*4*8*8, out_features=128)
        self.tnfc2 = nn.Linear(in_features=128, out_features=64)
        self.tnfc3 = nn.Linear(in_features=64, out_features=4)
        
        # Domain Classifier
        self.dcfc1 = nn.Linear(in_features=bf*4*8*8, out_features=64)
        self.dcfc2 = nn.Linear(in_features=64, out_features=32)
        self.dcfc3 = nn.Linear(in_features=32, out_features=1)
        
    def forward(self, x):
        x = x.view(-1, 3, 64, 64)
        # Feature Extractor
        x = self.dropout(self.pool(self.bn1(F.relu(self.conv1(x)))))
        x = self.dropout(self.pool(self.bn2(F.relu(self.conv2(x)))))
        x = self.dropout(self.pool(self.bn3(F.relu(self.conv3(x)))))
        
        x = x.view(-1, bf*4*8*8)
        
        # Task
        estimate = self.dropout(F.relu(self.tnfc1(x)))
        estimate = F.relu(self.tnfc2(estimate))
        estimate = F.softmax(self.tnfc3(estimate), dim=1)
        estimate = estimate.view(-1, 4)
        
        #Domain Classifier
        reverse_x = ReverseLayerF.apply(x)
        domain = F.relu(self.dcfc1(reverse_x))
        domain = F.relu(self.dcfc2(domain))
        domain = F.sigmoid(self.dcfc3(domain))
        domain = domain.view(-1)
        
        return estimate, domain