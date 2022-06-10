import os
from PIL import Image
import tqdm
import csv


datapath = os.path.join(os.getcwd(), 'data')
resizedpath = os.path.join(os.getcwd(), 'resized')

# Start off csv file
header = ['imagename', 'year', 'blok', 'club']
csvfilename = os.path.join(os.getcwd(), 'club_labels.csv')
f = open(csvfilename, 'w', encoding='UTF8', newline='')
csvwriter = csv.writer(f)
csvwriter.writerow(header)

# Resize all images for all years and put them into the "resized" folder
for year in os.listdir(datapath):
    yearpath = os.path.join(datapath, year)
    for blok in os.listdir(yearpath):
        blokpath = os.path.join(yearpath, blok)
        i = 1
        for imagename in tqdm.tqdm(os.listdir(blokpath), desc=year + ' ' + blok):
            imagepath = os.path.join(blokpath, imagename)
            image = Image.open(imagepath)
            new_image = image.resize((256, 256))
            # new_image.show()
            new_imagename = year + '_' + blok + '_' + str(i) + '.jpg'
            new_imagepath = os.path.join(resizedpath, new_imagename)
            new_image.save(new_imagepath)
            csvwriter.writerow([new_imagename, year, blok])
            i += 1

f.close()