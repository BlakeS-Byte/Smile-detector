# A program that uses CNN deep learning technique to detect if an image is smiling or not
#labels for data 
# 0->not happy
# 1-> happy

import numpy as np 
import pandas as pd                         #data processing of csv files
from sklearn.preprocessing import StandardScaler
import keras                                #for cnn
from tensorflow.python.keras.utils.np_utils import to_categorical    #I spent way to long trying to get this thing to work so here is this rediculous import for the categorical data
import tensorflow as tf                     #for image data
from numpy import genfromtxt

test_data = pd.read_csv('test.csv')
training_data = pd.read_csv('train.csv')

X_train = []
y_train = []
X_test = []
y_test = []

#split and normalize data
y_train = training_data.iloc[:,0:1]/255
y_test = test_data.iloc[:,0:1]/255
X_train = training_data.iloc[:,1:]/255
X_test = test_data.iloc[:,1:]/255

print(y_test)

#transfer data to a numpy array
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

#reshape the X_train and X_test data
X_train = X_train.reshape(X_train.shape[0],48,48,1)
X_test = X_test.reshape(X_test.shape[0],48,48,1)

#convert the label data to catagorical
y_train = to_categorical(y_train,num_classes=2,dtype = "int32")
y_test = to_categorical(y_test,num_classes=2, dtype= "int32")


#creating the convultional base CNN with tensor flow library takes in (image_height,image_width,color_channels)
model = models.Sequential()









