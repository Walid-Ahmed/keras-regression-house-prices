# USAGE
# python mlp_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import argparse
import locale
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True,help="path to input dataset of house images")
args = vars(ap.parse_args())

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
print("[INFO] loading house attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])


cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

# determine (1) the unique zip codes and (2) the number of data
# points with each zip code
#Pandas Index.value_counts() function returns object containing counts of unique values. The resulting object will be in descending order so that the first element is the most frequently-occurring element. Excludes NA values by default.
zipcodeSeries=df["zipcode"].value_counts()  #<class 'pandas.core.series.Series'>

zipcodes = zipcodeSeries.keys().tolist()   #zipcodes as list
counts = zipcodeSeries.tolist()    #count of zipcodes as list  



# loop over each of the unique zip codes and their corresponding
# count
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
	


column_names_to_normalize = ["bedrooms", "bathrooms", "area","price"]  #continous data
x = df[column_names_to_normalize].values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = df.index)
df[column_names_to_normalize] = df_temp
#print the first 5 lines
print(df.head())


df['zipcode']=pd.Categorical(df['zipcode'])
dfDummies = pd.get_dummies(df['zipcode'], prefix = 'zipcode')
print(dfDummies.head())
df = pd.concat([df, dfDummies], axis=1)
del df['zipcode']
print(df.head())





# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)


trainX=(train.drop('price', axis=1)).values
trainY=train["price"].values
testX=(test.drop('price', axis=1)).values
testY=test["price"].values

print(trainX[0,:])
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

exit()

#(trainX, testX) = datasets.process_house_attributes(df, train, test)

# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["price"].mean(), grouping=True),
	locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))