import pickle
import torch
import matplotlib.pyplot as plt

timeFile = open('Data/user_time.usrdata', 'rb')
time = pickle.load(timeFile)

heartFile = open('Data/user_heart.usrdata', 'rb')
heart = pickle.load(heartFile)

timeSickFile = open('Data/user_time_sick.usrdata', 'rb')
timeSick = pickle.load(timeSickFile)

sickFile = open('Data/user_sick.usrdata', 'rb')
sick = pickle.load(sickFile)

#print(time)
#print(heart)
#print(timeSick)
#print(sick)

sliceSize = 64
slices = []
sickSlices = []

for i in range(len(time)):
    for j in range(len(time[i])):
        for k in range(len(time[i][j])-sliceSize):
            sliceHeart = heart[i][j][k:k+sliceSize]
            if not 0 in sliceHeart:
                if timeSick[i][0] <= time[i][j][k+sliceSize] <= timeSick[i][-1]:
                    index = time[i][j][k+sliceSize]-timeSick[i][0]
                    wasSick = sick[i][index]

                    slices.append(torch.tensor(sliceHeart))
                    sickSlices.append(wasSick)

                    plt.plot(slices[-1])
    plt.xlabel("Day")
    plt.ylabel("Heart Rate")
    plt.show()

print("Total datapoints: "+str(len(slices)))

healthy = []
for i in range(len(slices)):
    if sickSlices[i] == 0:
        plt.plot(slices[i])
        healthy.append(slices[i].unsqueeze(0))
healthy = torch.cat(healthy, 0)
healthy = healthy[torch.randperm(healthy.shape[0])]
healthyTrain = healthy[:int(0.70*healthy.shape[0])]
healthyVal = healthy[int(0.7*healthy.shape[0]):int(0.85*healthy.shape[0])]
healthyTest = healthy[int(0.85*healthy.shape[0]):]
print(healthy)
print(healthy.shape)
print(healthyTrain.shape)
print(healthyVal.shape)
print(healthyTest.shape)


plt.title("Healthy")
plt.show()

sick = []
for i in range(len(slices)):
    if sickSlices[i] == 1:
        plt.plot(slices[i])
        sick.append(slices[i].unsqueeze(0))

sick = torch.cat(sick, 0)
sick = sick[torch.randperm(sick.shape[0])]
sickTrain = sick[:int(0.7*sick.shape[0])]
sickVal = sick[int(0.7*sick.shape[0]):int(0.85*sick.shape[0])]
sickTest = sick[int(0.85*sick.shape[0]):]
print(sick)
print(sick.shape)
print(sickTrain.shape)
print(sickVal.shape)
print(sickTest.shape)

plt.title("Sick")
plt.show()

fileSickTrain = open('Data/network_sick_train.data', 'wb')
pickle.dump(sickTrain,fileSickTrain)

fileSickVal = open('Data/network_sick_val.data', 'wb')
pickle.dump(sickVal,fileSickVal)

fileSickTest = open('Data/network_sick_test.data', 'wb')
pickle.dump(sickTest,fileSickTest)

fileHealthyTrain = open('Data/network_healthy_train.data', 'wb')
pickle.dump(healthyTrain,fileHealthyTrain)

fileHealthyVal = open('Data/network_healthy_val.data', 'wb')
pickle.dump(healthyVal,fileHealthyVal)

fileHealthyTest = open('Data/network_healthy_test.data', 'wb')
pickle.dump(healthyTest,fileHealthyTest)
