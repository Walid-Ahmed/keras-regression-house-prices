# USAGE
# python housePrice_regression.py 

# import the necessary packages
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import locale
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
import sys

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy
numpy.set_printoptions(threshold=sys.maxsize)



print("[INFO] loading house attributes...")
inputPath =  "HousesInfo.txt"
cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)
print(df.head())


#remove zipcounts that have kess than 25 houses
#Pandas Index.value_counts() function returns object containing counts of unique values. The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.
zipcodeSeries=df["zipcode"].value_counts()  #<class 'pandas.core.series.Series'>
zipcodes = zipcodeSeries.keys().tolist()   #zipcodes as list
counts = zipcodeSeries.tolist()    #count of zipcodes as list  
for (zipcode, count) in zip(zipcodes, counts):
		# the zip code counts for our housing dataset is *extremely*
		# unbalanced (some only having 1 or 2 houses per zip code)
		# so let's sanitize our data by removing any houses with less
		# than 25 houses per zip code
		if count < 25:
			booleanVal=(df["zipcode"] == zipcode)  # this will be true at all zipcodes that should be deleted
			#print(type(booleanVal))   #<class 'pandas.core.series.Series'>
			idxs = df[booleanVal].index  #this will return indices of these true values
			df.drop(idxs, inplace=True)
print("[INFO]removed zipcodes which less than 25 houses")            


#Normalize  continous values to be between 0 and 1
column_names_to_normalize = ["bedrooms", "bathrooms", "area"]  #continous data
x = df[column_names_to_normalize].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp
print("[INFO] Continous values normalize to be between 0 and 1")


#change categoital data to one hot vector
df['zipcode']=pd.Categorical(df['zipcode'])
dfDummies = pd.get_dummies(df['zipcode'], prefix = 'zipcode')
print(dfDummies.head())
df = pd.concat([df, dfDummies], axis=1)
df.drop(['zipcode'],axis=1,inplace=True)
print("[INFO] Zipcode converted to one hot vector")







# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)
trainX=(train.drop('price', axis=1)).values
trainY=train["price"].values
testX=(test.drop('price', axis=1)).values
testY=test["price"].values
print("[INFO]  train and test data prepared")


maxPrice = train["price"].max()
trainY=trainY/maxPrice
testY=testY/maxPrice
print("[INFO] Normalized price by printing by max price")






print("[INFO] trainX.shape  {}".format(trainX.shape))
print("[INFO]testX.shape {}".format(testX.shape))
print("[INFO] trainY.shape {}".format(trainY.shape))
print("[INFO] testY.shape {}".format(testY.shape))





model = Sequential()
model.add(Dense(8, input_dim=trainX.shape[1], activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear"))
model.summary()
plot_model(model, to_file='model.png')
import matplotlib.image as mpimg
img=mpimg.imread('model.png')
imgplot = plt.imshow(img)
plt.show()





opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
history=model.fit(trainX, trainY, validation_data=(testX, testY),epochs=200, batch_size=8)
model.save("housePrice.keras2")
print("[INFO] model saved to housePrice.keras2")

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)

validationLoss=(history.history['val_loss'])
trainingLoss=history.history['loss']




#------------------------------------------------
# Plot training and validation accuracy per epoch
epochs   = range(len(validationLoss)) # Get number of epochs
 #------------------------------------------------
plt.plot  ( epochs,     trainingLoss ,label="Training Loss")
plt.plot  ( epochs, validationLoss, label="Validation Loss" )
plt.title ('Training and validation loss')
plt.xlabel("Epoch #")
plt.ylabel("Loss")
fileToSaveAccuracyCurve="plot_acc.png"
plt.savefig("plot_acc.png")
print("[INFO] Loss curve saved to {}".format("plot_acc.png"))
plt.legend(loc="upper right")
plt.show()



#readjust house prices
testY=testY*maxPrice
preds=preds*maxPrice
#plot curves (Actual vs Predicted)
plt.plot  ( testY ,label="Actual price")
plt.plot  ( preds, label="Predicted price" )
plt.title ('House prices')
plt.xlabel("Point #")
plt.ylabel("Price")
plt.legend(loc="upper right")
plt.show()
plt.savefig("HousePrices.png")
print("[INFO] predicted vs actual price saved to HousePrices.png")


