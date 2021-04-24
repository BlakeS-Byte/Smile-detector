from numpy import loadtxt                                      # used to grab the data from csv files
from keras.models import Sequential                            # used for the model
from keras.layers import Dense                                 # used for the layers
import numpy as np                                             # used for arrays
from PIL import Image

# function prints out the first 5 images that are predicted to be smiles and are actually smiles
def print_pred(i):
    imt = X_test[i]
    image2 = Image.fromarray(np.reshape(imt, [48, 48]))
    image2.show()

dataset_train = loadtxt('train.csv', delimiter=',')            # take the training from the csv file
dataset_test = loadtxt('test.csv', delimiter=',')              # take the test data

X_train = dataset_train[:, 1:]                                 # the training data
X_test = dataset_test[:, 1:]                                   # the test data
y_train = dataset_train[:, 0:1]                                # labels for the train data
y_test = dataset_test[:, 0:1]                                  # labels for the test data

model = Sequential()
model.add(Dense(2500, input_dim=2304, activation='relu'))      # first layer
model.add(Dense(2000, activation='relu'))                      # second layer
model.add(Dense(1000, activation='relu'))                      # third layer
model.add(Dense(100, activation='relu'))                       # fourth layer
model.add(Dense(1, activation='sigmoid'))                      # fifth layer

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # combine the layers into full model
model.fit(X_train, y_train, epochs=15, batch_size=1000)                             # fit the training data to the model

model.summary()

predictions = (model.predict(X_test) > 0.5).astype("int32")    # using the test data, predicts the labels
accuracy = sum(np.array(predictions == y_test)) / len(X_test)  # checks the accuracy of the predictions
print(accuracy * 100)                                          # prints out the accuracy

count = 0                                                      # used to print a max of 5 images
for i in range(len(X_test)):                                   # loops until 5 have been printed
    if count == 5:                                             # breaks at 5
        break
    if predictions[i] == 1 and y_test[i] == 1:                 # checks for predicted true and actual true
        print_pred(i)
        count += 1
