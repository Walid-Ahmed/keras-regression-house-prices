# keras-regression-house-prices

Special thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/)   for his [great post](https://www.pyimagesearch.com/2019/01/21/regression-with-keras/) 

This is simple code  creates and train a neural network to predict house prices

usage 'python  housePrice_regression.py'



The model used is as the following:

<img src="https://github.com/Walid-Ahmed/keras-regression-house-prices/blob/master/sampleImages/model.png"  align="middle">

The dataset is from   https://github.com/emanhamed/Houses-dataset, the house dataset includes four numerical and categorical attributes as input and the one continous variable as output:
1. Number of bedrooms (continous)
2. Number of bathrooms(continous)
3. Area (continous)
4. Zip code (Cateogiral)
5.Price (continous)

The variable to precit is the price of the house

When training finishes the following, curves shown the traning and validation  loss is shown. Another curce will also be shown for actualvs predicted prices. Both curves are saved to local drive

<img src="https://github.com/Walid-Ahmed/keras-regression-house-prices/blob/master/sampleImages/loss.png">

<img src="https://github.com/Walid-Ahmed/keras-regression-house-prices/blob/master/sampleImages/price.png">

