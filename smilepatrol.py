# A program that uses CNN deep learning technique to detect if an image is smiling or not
#labels for data 
# 0->not happy
# 1-> happy

import numpy as np
from numpy import genfromtxt
import pandas as pd                         #data processing of csv files
import keras                                #for cnn
import tensorflow as tf                     #for image data
from tensorflow.python.keras.utils.np_utils import to_categorical    #I spent way to long trying to get this thing to work so here is this rediculous import for the categorical data
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt             #plotting data as image
from sklearn.preprocessing import StandardScaler


test_data = pd.read_csv('test.csv')
training_data = pd.read_csv('train.csv')

X_train = []
y_train = []
X_test = []
y_test = []

#split and normalize data
y_train = training_data.iloc[:,0:1]
y_test = test_data.iloc[:,0:1]
X_train = training_data.iloc[:,1:]/255
X_test = test_data.iloc[:,1:]/255

#print(y_test)

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

#Plots the first 9 images in X_train
#plt.figure(figsize=(5,5))
#for i in range(9):
#    plt.subplot(3,3,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(X_train[i], cmap=plt.cm.binary)
#    plt.xlabel(y_train[i])
#plt.show()

#creating the convultional base CNN with tensor flow library takes in (image_height,image_width,color_channels)
model = models.Sequential()









