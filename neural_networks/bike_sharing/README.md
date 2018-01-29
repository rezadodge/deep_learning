# <span style="color:darkred"> Predicting Bike Sharing Demand using Neural Network </span>


A neural network model is built and trained to predict bike sharing demand in Washington, DC. The model is a neural network with one hidden layer, written in Python using `NumPy` and `Pandas`. The training dataset contains the hourly and daily count of rental bikes in 2011 and 2012 in Capital Bikeshare system with the corresponding weather and seasonal information. ([Link to dataset folder](https://github.com/rezadodge/deep_learning/tree/master/neural_networks/bike_sharing/Bike-Sharing-Dataset); [Link to the repository on UCI Machine Learning Database](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset))

### <span style="color:darkred"> Predictions for the last three weeks of the year </span>
The predictions are very good up to December 21st, which is the last working day before Christmas, which was on Tuesday in year 2012. Considering the fact that the training data included only one Christmas (for the year 2011), it is not surprising that the model predictions are not very good for this week.

<img src="assets/prediction01.png" width=900px>
