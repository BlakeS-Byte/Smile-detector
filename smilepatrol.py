# A program that uses CNN deep learning technique to detect if an image is smiling or not
import numpy as np 
import pandas as pd                         #data processing of csv files
from sklearn.preprocessing import StandardScaler
import keras                                #for cnn
import tensorflow as tf                     #for image data

train_data = "../input/emotion-detection-fer/train/"        #training images 
test_data = "../input/emotion-detection-fer/test/"          #test images

