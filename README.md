# QuantFluNNAnalysis
This is my code for data processing and training neural networks and other ML models for the QuantifiedFlu project. Below is an overview of what the four key files do.

## GetData.py
This file retreives up-to-date heartrate and sickness data from the QuantifiedFlu website.

## DataToNetwork.py
This file converts the heartrate and sickness data into a format useful for a neural network (i.e. it creates data slices).

## DataTraining.py
This file can be used to train either a neural network or tree model on the data. It uses the following two files:

### Network.py
This file contains the code for training a neural network on the data. Currently, this is only a simple proof-of-concept fully-connected neural network.

### XGBoost.py
This file contains the code for interfacing with the xgboost package to train and Extreme Gradient Tree Boosting model on the data.
