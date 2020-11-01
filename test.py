import pickle
import torch
import matplotlib.pyplot as plt

def fix1(times,heartRates):
    print(times)
    print(heartRates)
    startTime = times[0]

    for i in range(1,len(times)):
        times[i] = (times[i]-times[0]).days
    times[0] = (times[0]-times[0]).days
    times = torch.tensor(times)
    heartRates = torch.tensor(heartRates)

    times2 = torch.arange(0,torch.max(times)+1,1)
    heartRates2 = []

    for i in range(times2.shape[0]):
        mask = (times == times2[i])
        if not mask.any():
            heartRates2.append(0)
        else:
            mask2 = (heartRates[mask] < torch.mean(heartRates[mask])-100*torch.std(heartRates[mask]))
            if torch.sum(mask2)<10:
                heartRates2.append(torch.mean(torch.sort(heartRates[mask])[0][0:1]))
            else:
                heartRates2.append(torch.mean(heartRates[mask][mask2]))
    return (startTime,times2,heartRates2)

def fix2(times,heartRates):
    print(times)
    print(heartRates)
    startTime = times[0]

    for i in range(1,len(times)):
        times[i] = (times[i]-times[0]).days
    times[0] = (times[0]-times[0]).days
    times = torch.tensor(times)
    heartRates = torch.tensor(heartRates)

    times2 = torch.arange(0,torch.max(times)+1,1)
    heartRates2 = []

    for i in range(times2.shape[0]):
        mask = (times == times2[i])
        if not mask.any():
            heartRates2.append(0)
        else:
            mask2 = (heartRates[mask] < torch.mean(heartRates[mask])-100*torch.std(heartRates[mask]))
            if torch.sum(mask2)<10:
                heartRates2.append(torch.median(torch.sort(heartRates[mask])[0][0:1]))
            else:
                heartRates2.append(torch.median(heartRates[mask][mask2]))
    return (startTime,times2,heartRates2)

def fixSum(times,heartRates):
    print(times)
    print(heartRates)
    startTime = times[0]

    for i in range(1,len(times)):
        times[i] = (times[i]-times[0]).days
    times[0] = (times[0]-times[0]).days
    times = torch.tensor(times)
    heartRates = torch.tensor(heartRates)

    times2 = torch.arange(0,torch.max(times)+1,1)
    heartRates2 = []

    for i in range(times2.shape[0]):
        mask = (times == times2[i])
        if not mask.any():
            heartRates2.append(0)
        else:
            heartRates2.append(torch.mean(heartRates[mask]))
    return (startTime,times2,heartRates2)


fileTime5 = open('heart_5min_time.list', 'rb')
time5min = pickle.load(fileTime5)

fileTimeSum = open('heart_sum_time.list', 'rb')
timeSum = pickle.load(fileTimeSum)

fileHeart5 = open('heart_5min.list', 'rb')
heart5min = pickle.load(fileHeart5)

fileHeartSum = open('heart_sum.list', 'rb')
heartSum = pickle.load(fileHeartSum)

start5min,time5min1,heart5min1 = fix1(time5min.copy(),heart5min.copy())
start5min,time5min2,heart5min2 = fix2(time5min,heart5min)
startSum,timeSum,heartSum = fixSum(timeSum,heartSum)

startDiff = (start5min-startSum).days
for i in range(len(time5min)):
    time5min[i] += startDiff

plt.plot(time5min1,heart5min1,label="5min mean")
plt.plot(time5min2,heart5min2,label="5min median")
plt.plot(timeSum,heartSum,label="summary")
plt.legend()
plt.show()


