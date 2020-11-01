import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import pickle

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.fc1 = nn.Linear(64,30)
        self.fc2 = nn.Linear(30,16)#20
        self.fc3 = nn.Linear(16,10)
        self.fc4 = nn.Linear(10,6)
        self.fc5 = nn.Linear(6,3)
        self.fc6 = nn.Linear(3,1)
        self.bestModel = None

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=torch.sigmoid(self.fc6(x))
        return x

    def sampleData(self, x1, x2, batchSize):
        rand1 = torch.randperm(x1.shape[0])[0:batchSize]
        rand2 = torch.randperm(x2.shape[0])[0:batchSize]

        xbatch = torch.cat((x1[rand1,...],x2[rand2,...]),0)
        return (xbatch + (torch.rand(xbatch.shape)-0.5)/1.5)
    
    def train(self,x1,x2,x1val,x2val,batchSize,learningRate,momentum,numEpochs):
        valComp = torch.cat([torch.zeros([x1val.shape[0],1]),torch.ones([x2val.shape[0],1])],0)
        ybatch = torch.cat((torch.zeros([batchSize,1]),torch.ones([batchSize,1])),0)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learningRate)#, momentum=momentum)
        losses = []
        valLosses = []
        accuracies = []
        accuraciesVal = []
        maxAcc = 0
        for i in range(1,numEpochs+1):
            avgLoss = 0
            avgValLoss = 0
            avgAcc = 0
            avgAccVal = 0
            for j in range(int((x1.shape[0]+x2.shape[0])/batchSize)):
                xbatch = self.sampleData(x1,x2,batchSize)
                #print(xbatch.shape)
                #print(ybatch.shape)

                optimizer.zero_grad()
                output = self.forward(xbatch)
                loss = criterion(output, ybatch)
                loss.backward()
                optimizer.step()

                outRound = torch.round(output)
                outCorrect = torch.sum(torch.abs(outRound-ybatch))
                acc = 1-(outCorrect/output.shape[0]).item()
                avgAcc += acc

                valOut = self.forward(torch.cat([x1val,x2val],0))
                valLoss = criterion(valOut, valComp)
                avgLoss += loss.item()
                avgValLoss += valLoss.item()

                outValRound = torch.round(valOut)
                outValCorrect = torch.sum(torch.abs(outValRound-valComp))
                accVal = 1-(outValCorrect/valComp.shape[0]).item()
                avgAccVal += accVal

            avgLoss /= int((x1.shape[0]+x2.shape[0])/batchSize)
            avgValLoss /= int((x1.shape[0]+x2.shape[0])/batchSize)
            avgAcc /= int((x1.shape[0]+x2.shape[0])/batchSize)
            avgAccVal /= int((x1.shape[0]+x2.shape[0])/batchSize)

            if avgAccVal*avgAcc > maxAcc:
                maxAcc = avgAccVal*avgAcc
                self.bestModel = self
            
            losses.append(avgLoss)
            valLosses.append(avgValLoss)
            accuracies.append(avgAcc)
            accuraciesVal.append(avgAccVal)

            if i%10 == 0:
                print(i)
                print(avgLoss)
                print(avgAcc)
                print(avgAccVal)

        valOut = torch.round(self.bestModel.forward(torch.cat([x1val,x2val],0)))
        valHealthy = valOut[:x1val.shape[0]]
        valSick = valOut[x1val.shape[0]:]
        
        valHealthyCorrect = valHealthy.shape[0]-torch.sum(valHealthy)
        valHealthyIncorrect = torch.sum(valHealthy)

        valSickCorrect = torch.sum(valSick)
        valSickIncorrect = valSick.shape[0]-torch.sum(valSick)

        print("Healthy Correct")
        print(valHealthyCorrect)
        print("Healthy Incorrect")
        print(valHealthyIncorrect)
        print("Sick Correct")
        print(valSickCorrect)
        print("Sick Incorrect")
        print(valSickIncorrect)

        plt.plot(losses,label = "Training Loss")
        plt.plot(valLosses,label = "Validation Loss")
        plt.plot(accuracies,label = "Training Accuracy")
        plt.plot(accuraciesVal,label = "Validation Accuracy")
        plt.legend()
        plt.show()


sickFileTrain = open('network_sick_train.data', 'rb')
sickTrain = pickle.load(sickFileTrain)

sickFileVal = open('network_sick_val.data', 'rb')
sickVal = pickle.load(sickFileVal)

sickFileTest = open('network_sick_test.data', 'rb')
sickTest = pickle.load(sickFileTest)

healthyFileTrain = open('network_healthy_train.data', 'rb')
healthyTrain = pickle.load(healthyFileTrain)

healthyFileVal = open('network_healthy_val.data', 'rb')
healthyVal = pickle.load(healthyFileVal)

healthyFileTest = open('network_healthy_test.data', 'rb')
healthyTest = pickle.load(healthyFileTest)

print(sickTrain.shape)
print(sickVal.shape)
print(sickTest.shape)
print(healthyTrain.shape)
print(healthyVal.shape)
print(healthyTest.shape)

n = Network()
n.train(healthyTrain,sickTrain,healthyVal,sickVal,8,0.0001,0.9,1000)
