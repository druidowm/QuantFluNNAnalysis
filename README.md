# QuantFluNNAnalysis
This is my code for data processing and neural network training for the QuantifiedFlu project. Below is an overview of what the three key files do.

## GetData.py
This file retreives up-to-date heartrate and sickness data from the QuantifiedFlu website.

## DataToNetwork.py
This file converts the heartrate and sickness data into a format useful for a neural network (i.e. it creates data slices).

## Network.py
This file trains an actual neural network. Currently, this is only a simple proof-of-concept fully-connected neural network.
