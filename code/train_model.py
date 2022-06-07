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

is_laga = club_labels["club"] == "laga" # PandasArray of True and False
is_laga_int = [int(is_laga_instance) for is_laga_instance in is_laga] # List of 0 and 1
is_laga_one_hot = [(1, 0) if i else (0, 1) for i in is_laga] # List of (1, 0) and (0, 1)
file_names = club_labels["imagename"]


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