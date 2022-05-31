import os
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


resizedpath = os.path.join(os.getcwd(), 'resized')
csvfilename = os.path.join(os.getcwd(), 'club_labels.csv')

club_labels = pd.read_csv(csvfilename)
print(club_labels)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(3*100*100, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()