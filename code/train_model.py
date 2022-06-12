import os
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Variables
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

resizedpath = os.path.join(os.getcwd(), 'resized')
csvfilename = os.path.join(os.getcwd(), 'club_labels.csv')


# Code
club_labels = pd.read_csv(csvfilename, sep=";")
print(club_labels)

all_clubs = set(club_labels["club"])
is_laga = club_labels["club"] == "laga" # PandasArray of True and False
is_laga_int = [int(is_laga_instance) for is_laga_instance in is_laga] # List of 0 and 1
is_laga_one_hot = [(1, 0) if i else (0, 1) for i in is_laga] # List of (1, 0) and (0, 1)
file_names = club_labels["imagename"]


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Number of classes in the dataset
num_classes = 2
# Import model
resnet18 = models.resnet18(pretrained=True)
set_parameter_requires_grad(resnet18, feature_extract)
# Edit last layer to predict suit our model
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, num_classes)
input_size = 224
