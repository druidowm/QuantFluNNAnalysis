import pickle

from Network import Network
from XGBoost import XGBobj

import torch

import numpy as np

def getData():
    sickFileTrain = open('Data/network_sick_train.data', 'rb')
    sickTrain = pickle.load(sickFileTrain)

    sickFileVal = open('Data/network_sick_val.data', 'rb')
    sickVal = pickle.load(sickFileVal)

    sickFileTest = open('Data/network_sick_test.data', 'rb')
    sickTest = pickle.load(sickFileTest)

    healthyFileTrain = open('Data/network_healthy_train.data', 'rb')
    healthyTrain = pickle.load(healthyFileTrain)

    healthyFileVal = open('Data/network_healthy_val.data', 'rb')
    healthyVal = pickle.load(healthyFileVal)

    healthyFileTest = open('Data/network_healthy_test.data', 'rb')
    healthyTest = pickle.load(healthyFileTest)

    print(sickTrain.shape)
    print(sickVal.shape)
    print(sickTest.shape)
    print(healthyTrain.shape)
    print(healthyVal.shape)
    print(healthyTest.shape)

    return (sickTrain,sickVal,sickTest,healthyTrain,healthyVal,healthyTest)

def duplicateRandom(data, numDuplicates):
    originalData = data
    for _ in range(numDuplicates):
        data = torch.cat((data, originalData+(torch.rand(originalData.shape)-0.5)/1.5),0)
    return data

def augmentData(sickTrain, healthyTrain, numDuplicates):
    scaleAmount = int(healthyTrain.shape[0]/sickTrain.shape[0])
    sickTrain = duplicateRandom(sickTrain, numDuplicates*scaleAmount)
    healthyTrain = duplicateRandom(healthyTrain, numDuplicates)
    return (sickTrain,healthyTrain)

def NNTrain():
    sickTrain,sickVal,sickTest,healthyTrain,healthyVal,healthyTest = getData()
    n = Network()
    n.train(healthyTrain,sickTrain,healthyVal,sickVal,8,0.0001,0.9,1000)

def XGBTrain():
    sickTrain,sickVal,sickTest,healthyTrain,healthyVal,healthyTest = getData()
    sickTrain,healthyTrain = augmentData(sickTrain,healthyTrain, 10)
    x = XGBobj()

    train_X = torch.cat((sickTrain,healthyTrain),dim = 0)
    val_X = torch.cat((sickVal,healthyVal),dim = 0)
    test_X = torch.cat((sickTest,healthyTest),dim = 0)

    train_Y = torch.cat((torch.ones([sickTrain.shape[0]]),torch.zeros([healthyTrain.shape[0]])), dim=0)
    val_Y = torch.cat((torch.ones([sickVal.shape[0]]),torch.zeros([healthyVal.shape[0]])), dim=0)
    test_Y = torch.cat((torch.ones([sickTest.shape[0]]),torch.zeros([healthyTest.shape[0]])), dim=0)
    print(train_Y)
    print(val_Y)
    print(test_Y)
    x.train(train_X.numpy(),train_Y.numpy(),val_X.numpy(),val_Y.numpy(),10000)

    x.test(val_X.numpy(),val_Y.numpy())

XGBTrain()