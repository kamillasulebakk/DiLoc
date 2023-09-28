import torch.nn as nn
import torch.nn.functional as F
import torch

def number_of_output_values(amp: bool, area: bool):
    result = 3
    if amp:
        result += 1
    if area:
        result += 1
    return result

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Size of input image 20x20x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # Size: 16x16x6
        self.pool = nn.MaxPool2d(2, stride=1)
        # Size: 15x15x6
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Size: 11x11x16
        # Size after pool: 10x10x16
        self.fc1 = nn.Linear(10*10*16, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 3)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x