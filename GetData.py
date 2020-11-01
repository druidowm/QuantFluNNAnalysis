import urllib.request
import json
from datetime import date,datetime
from dateutil import parser
import torch
import matplotlib.pyplot as plt
import ssl
import pickle

ssl._create_default_https_context = ssl._create_unverified_context

def readInternetJson(link):
    with urllib.request.urlopen(link) as url:
        data = json.loads(url.read().decode())
        return data


def readHeartRate(data,shift=False):
    times = []
    heartRates = []
    for item in data:
        heartRate = item["data"]["heart_rate"]
        if heartRate != "_" and heartRate != "-":
            heartRates.append(float(heartRate))
            timestamp = parser.isoparse(item["timestamp"])
            timestamp = timestamp.date()
            #if timestamp.hour >= 11:
            #    timestamp = timestamp.date()
            #    try:
            #        timestamp.replace(day = timestamp.day+1)
            #    except:
            #        try:
            #            timestamp.replace(month = timestamp.month+1, day = 1)
            #        except:
            #            timestamp.replace(year = timestamp.year+1, month = 1, day = 1)
            #else:
            #    timestamp = timestamp.date()
            times.append(timestamp)
    #print(times)

    if len(times)==0:
        return (0,times,heartRates)

    startTime = times[0]

    for i in range(1,len(times)):
        times[i] = (times[i]-times[0]).days
    times[0] = (times[0]-times[0]).days
    times = torch.tensor(times)
    #print(times)
    heartRates = torch.tensor(heartRates)

    times2 = torch.arange(0,torch.max(times)+1,1)
    heartRates2 = []

    for i in range(times2.shape[0]):
        mask = (times == times2[i])
        if not mask.any():
            heartRates2.append(0)
        else:
            heartRates2.append(torch.min(heartRates[mask]))

    if shift:
        times2 -= 1

    #print("startTime")
    #print(startTime)

    return (startTime, times2,heartRates2)

def getSick(symptoms):
    times = []
    sick = []
    for item in symptoms:
        if len(item["data"])>0:
            symptoms = 0
            if "symptom_anosmia" in item["data"]:
                symptoms += item["data"]["symptom_anosmia"]
            if "symptom_body_ache" in item["data"]:
                symptoms += item["data"]["symptom_body_ache"]
            if "symptom_chills" in item["data"]:
                symptoms += item["data"]["symptom_chills"]
            if "symptom_cough" in item["data"]:
                symptoms += item["data"]["symptom_cough"]
            if "symptom_diarrhea" in item["data"]:
                symptoms += item["data"]["symptom_diarrhea"]
            if "symptom_ear_ache" in item["data"]:
                symptoms += item["data"]["symptom_ear_ache"]
            if "symptom_fatigue" in item["data"]:
                symptoms += item["data"]["symptom_fatigue"]
            if "symptom_headache" in item["data"]:
                symptoms += item["data"]["symptom_headache"]
            if "symptom_nausea" in item["data"]:
                symptoms += item["data"]["symptom_nausea"]
            if "symptom_runny_nose" in item["data"]:
                symptoms += item["data"]["symptom_runny_nose"]
            if "symptom_short_breath" in item["data"]:
                symptoms += item["data"]["symptom_short_breath"]
            if "symptom_sore_throat" in item["data"]:
                symptoms += item["data"]["symptom_sore_throat"]
            if "symptom_stomach_ache" in item["data"]:
                symptoms += item["data"]["symptom_stomach_ache"]
            if "fever" in item["data"] and isinstance(item["data"]["fever"],int) and item["data"]["fever"]>=99:
                symptoms += item["data"]["fever"]-99

            if symptoms>0:
                sick.append(1)
            else:
                sick.append(0)

            times.append(parser.isoparse(item["timestamp"]).date())

    if len(times)==0:
        return (0,times,sick)


    startTime = times[0]

    for i in range(1,len(times)):
        times[i] = (times[i]-times[0]).days
    times[0] = (times[0]-times[0]).days

    #plt.plot(times,sick)
    #plt.show()

    times = torch.tensor(times)
    sick = torch.tensor(sick)

    times2 = torch.arange(0,torch.max(times)+1,1)
    sick2 = []

    for i in range(times2.shape[0]):
        mask = (times == times2[i])
        if not mask.any():
            sick2.append("NA")
        else:
            sick2.append(sick[mask][0])
            j = len(sick2)-2
            if sick2[j] == 1:
                if sick2[j+1] == "NA":
                    sick2[j+1] = 1
            while sick2[j] == "NA":
                if sick2[-1] == 1:
                    if j == len(sick2)-2:
                        sick2[j] = 1
                    else:
                        sick2[j] = 0
                else:
                    sick2[j] = 0

                j -= 1
        #print("sick2")
        #print(sick2)

    sick2 = torch.tensor(sick2)

    #plt.plot(times2,sick2)
    #plt.show()

    return (startTime,times2,sick2)

def getData(links):
    metaData = [readInternetJson(link) for link in links]
    userTimes = []
    userHearts = []
    newSickTimes = []
    newSick = []

    for item in metaData:
        index = 0
        for item2 in item:
            #print("hi")
            #print(index)
            heartTimes = []
            sickTimes = []
            sick = []
            heartRates = []
            heartStart = []
            sickStart = None

            newTimes = []
            newHearts = []

            heartMonitor = []

            data = readInternetJson("https://quantifiedflu.org"+item2["json_path"])
            print(item2["member_id"])
            for point in data:
                if point == "fitbit_intraday" or point == "fitbit_summary" or point == "apple_health_summary" or point == "garmin_heartrate" or point == "googlefit_heartrate" or point == "oura_sleep_5min" or point == "oura_sleep_summary":
                    if data[point] != None:
                        if point == "oura_sleep_5min" or point == "apple_health_summary":
                            newHeartStart, newHeartTimes, newHeartRates = readHeartRate(data[point], True)
                        else:
                            newHeartStart, newHeartTimes, newHeartRates = readHeartRate(data[point])

                        if newHeartStart != 0:
                            heartStart.append(newHeartStart)
                            heartTimes.append(newHeartTimes)
                            heartRates.append(newHeartRates)

                            heartMonitor.append(point)

                elif point == "symptom_report":
                    sickStart, sickTimes,sick = getSick(data[point])

                else:
                    print("here")
                    print(point)

            if len(heartTimes) > 0 and len(sickTimes) > 0:
                timeDiff = [round((heartStart[i]-sickStart).total_seconds()/86400.0) for i in range(len(heartStart))]
                for i in range(len(heartTimes)):
                    for j in range(len(heartTimes[i])):
                        heartTimes[i][j]+=timeDiff[i]

                    plt.plot(heartTimes[i], heartRates[i], label = heartMonitor[i])

                newSickTimes.append(sickTimes)
                newSick.append(sick)

                if "fitbit_intraday" in heartMonitor and "fitbit_summary" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "fitbit_intraday":
                            index1 = i
                        if heartMonitor[i] == "fitbit_summary":
                            index2 = i
                    
                    fitbitTimes = []
                    fitbitHeart = []
                    for i in range(min(min(heartTimes[index1]),min(heartTimes[index2])),max(max(heartTimes[index1]),max(heartTimes[index2]))+1):
                        fitbitTimes.append(i)
                        if i in heartTimes[index1] and heartRates[index1][i-heartTimes[index1][0]] != 0:
                            if i in heartTimes[index2] and heartRates[index2][i-heartTimes[index2][0]] != 0:
                                fitbitHeart.append((heartRates[index1][i-heartTimes[index1][0]]+heartRates[index2][i-heartTimes[index2][0]])/2)
                            else:
                                fitbitHeart.append(heartRates[index1][i-heartTimes[index1][0]])
                        else:
                            fitbitHeart.append(heartRates[index2][i-heartTimes[index2][0]])

                    print("fitbit")
                    print(fitbitTimes)
                    print(fitbitHeart)

                    newTimes.append(fitbitTimes)
                    newHearts.append(fitbitHeart)

                elif "fitbit_intraday" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "fitbit_intraday":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])
                
                elif "fitbit_summary" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "fitbit_summary":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])
                
                if "oura_sleep_5min" in heartMonitor and "oura_sleep_summary" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "oura_sleep_5min":
                            index1 = i
                        if heartMonitor[i] == "oura_sleep_summary":
                            index2 = i
                    
                    ouraTimes = []
                    ouraHeart = []
                    for i in range(min(min(heartTimes[index1]),min(heartTimes[index2])),max(max(heartTimes[index1]),max(heartTimes[index2]))+1):
                        ouraTimes.append(i)
                        print(heartRates[index2][i-heartTimes[index2][0]])
                        if i in heartTimes[index1] and heartRates[index1][i-heartTimes[index1][0]] != 0:
                            if i in heartTimes[index2] and heartRates[index2][i-heartTimes[index2][0]] != 0:
                                ouraHeart.append((heartRates[index1][i-heartTimes[index1][0]]+heartRates[index2][i-heartTimes[index2][0]])/2)
                            else:
                                ouraHeart.append(heartRates[index1][i-heartTimes[index1][0]])
                        else:
                            ouraHeart.append(heartRates[index2][i-heartTimes[index2][0]])

                    #print("oura")
                    #print(ouraTimes)
                    #print(ouraHeart)
                    
                    newTimes.append(ouraTimes)
                    newHearts.append(ouraHeart)

                elif "oura_sleep_5min" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "oura_sleep_5min":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])
                
                elif "oura_sleep_summary" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "oura_sleep_summary":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])
                
                if "apple_health_summary" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "apple_health_summary":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])

                if "garmin_heartrate" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "garmin_heartrate":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])

                if "googlefit_heartrate" in heartMonitor:
                    for i in range(len(heartMonitor)):
                        if heartMonitor[i] == "googlefit_heartrate":
                            newTimes.append(heartTimes[i])
                            newHearts.append(heartRates[i])

                #print(newTimes)
                #print(newHearts)

                userTimes.append(newTimes)
                userHearts.append(newHearts)

                #for i in range(len(newTimes)):
                #    plt.plot(newTimes[i], newHearts[i], label = "new"+str(i+1))

                #plt.plot(sickTimes, sick, label = "Sick")
                plt.legend()
                plt.title("Member "+str(item2["member_id"]))
                plt.xlabel("Day")
                plt.ylabel("Heart Rate")
                plt.show()
                #for i in range(len(heartTimes)-16):


            index += 1
    
    fileTime = open('user_time.usrdata', 'wb')
    pickle.dump(userTimes,fileTime)

    fileHeart = open('user_heart.usrdata', 'wb')
    pickle.dump(userHearts,fileHeart)

    fileTimeSick = open('user_time_sick.usrdata', 'wb')
    pickle.dump(newSickTimes,fileTimeSick)

    fileSick = open('user_sick.usrdata', 'wb')
    pickle.dump(newSick,fileSick)

#getSick(readInternetJson("https://quantifiedflu.org/report/list/member/05321037.json")["symptom_report"])
getData(["https://quantifiedflu.org/report/public.json"])