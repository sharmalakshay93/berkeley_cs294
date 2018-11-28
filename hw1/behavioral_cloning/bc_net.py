import torch.nn as nn
import torch.nn.functional as F

class BCNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        