#Program for reading and transferring data to csv files from  This dataset contains https://www.kaggle.com/ananthu017/emotion-detection-fer
# which contains 35,685 examples of 48x48 pixel gray scale images of faces divided into 
# train and test dataset. Images are categorized based on the emotion shown 
# in the facial expressions (happiness, neutral, sadness, anger, surprise, disgust, fear).


import numpy as np
from PIL import Image
from numpy import asarray
import glob
import csv
import pandas

# format the train data
image_list = []
for filename in glob.glob('train/*/*.png'):
    im = Image.open(filename)
    data = asarray(im).flatten()
    image_list.append(data)

train_number_happy = len(glob.glob('train/happy/*'))
train_number_not_happy = len(glob.glob('train/angry/*')) + len(glob.glob('train/disgusted/*')) \
                   + len(glob.glob('train/fearful/*'))
pd = pandas.DataFrame(np.c_[np.zeros(len(image_list)), image_list])
pd.loc[train_number_not_happy+1:train_number_not_happy + train_number_happy, 0] = 1
pd.to_csv("train.csv")

# format the test data
image_list = []
for filename in glob.glob('test/*/*.png'):
    im = Image.open(filename)
    data = asarray(im).flatten()
    image_list.append(data)

test_number_happy = len(glob.glob('test/happy/*'))
test_number_not_happy = len(glob.glob('test/angry/*')) + len(glob.glob('test/disgusted/*')) \
                   + len(glob.glob('test/fearful/*'))
pd = pandas.DataFrame(np.c_[np.zeros(len(image_list)), image_list])
pd.loc[test_number_not_happy+1:test_number_not_happy + test_number_happy, 0] = 1
pd.to_csv("test.csv")