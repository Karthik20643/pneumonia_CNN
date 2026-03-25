import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):

    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 54 * 54)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x