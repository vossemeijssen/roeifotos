import csv
import os
import pandas as pd
import random
import shutil


resized_path = os.path.join(os.getcwd(), 'resized')
train_path = os.path.join(resized_path, 'train')
val_path = os.path.join(resized_path, 'val')
csv_filename = os.path.join(os.getcwd(), 'club_labels.csv')


def move_files(destination_folder, source_folder, list_of_files):
    for file in list_of_files:
        source = os.path.join(source_folder, file)
        destination = os.path.join(destination_folder, file)
        shutil.move(source, destination)

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

club_labels = pd.read_csv(csv_filename, sep=";")

# Check how many times each club occurs
all_clubs = set(club_labels["club"])
too_small_clubs = []
for club in all_clubs:
    n_images_for_this_club = sum(club_labels.club == club)
    print(f"{n_images_for_this_club} images from {club}")
    if n_images_for_this_club < 10:
        too_small_clubs.append(club)


all_non_unclear_images = []
all_unclear_images = []
for ind, row in club_labels.iterrows():
    if row.club not in too_small_clubs and row.club != "unclear":
        imagepath = row.imagename
        all_non_unclear_images.append(imagepath)


random.shuffle(all_non_unclear_images)

# 80% train, 20% validation
split_point = round(len(all_non_unclear_images) * 0.8)

train_images = all_non_unclear_images[:split_point]
val_images = all_non_unclear_images[split_point:]

move_files(train_path, resized_path, train_images)
move_files(val_path, resized_path, val_images)

