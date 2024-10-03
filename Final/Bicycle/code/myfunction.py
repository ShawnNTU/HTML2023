import numpy as np
import pandas as pd

from datetime import date
import datetime
import pickle

#===================== Model Saving and Loading =====================

def saveModel(model,name):
    with open(name,'wb') as F:
        pickle.dump(model,F)

def loadModel(name):
    with open(name,'rb') as F:
        return pickle.load(F)
    
#===================== Data Loader =====================
def myDataloaderCertainDay(sta, day):
    try:
        data = np.genfromtxt(f"./processed data/{sta}/{day}",dtype=int)
    except FileNotFoundError:
        return "FileNotFound"
    return data

def myDataloader(sta,days_interval:list):
    # sta:station id
    # days_interval: the time interval of data,
    # a list like ["20231201","20231202",...]
    # return a list of seven "day_list"
    # each day_list is the data of that day (e.g. Monday)
    # 0 is Sunday, 1 is Monday .etc
    cal = [[],[],[],[],[],[],[]]
    for day in days_interval:
        try:
            data = np.genfromtxt(f"./processed data/{sta}/{day}",dtype=int)
            cal[data[0][3]].append(data)
        except FileNotFoundError:
            return "FileNotFound"
    for i in range(7):
        if cal[i] != []:
            cal[i] = np.stack(cal[i],axis=0)
    return cal    

#===================== Time Conversion =====================

# date example: 20231021_500101001_00:00

def dateConversion(date_str:str):
    MONTH_VALUE = {10:6,11:2,12:4}
    month = int(date_str[4:6])
    date = int(date_str[6:8])
    the_day = (date + MONTH_VALUE[month]) % 7
    hour = int(date_str[-5:-3])
    minute = int(date_str[-2:])
    return (the_day, hour, minute)

def timeCalculator(time:str, delta):
    year = int(time[:4])
    month = int(time[4:6])
    day = int(time[6:])
    date_obj = date(year,month,day) - datetime.timedelta(delta)
    return f"{date_obj.year:04d}{date_obj.month:02d}{date_obj.day:02d}"
    

#===================== Data Generating about time series =====================

# index of sbi is -3, tot is -4
def dataCombing(current_day, next_week_day):
    # both of them are (1440,10) ndarray
    X = []
    Y = []
    Y_tot = []
    for i in range(20,1441): # don't forget it's end at 1441 not 1440
        X.append(current_day[i-20:i,-3]) # x_sbi
        Y.append(next_week_day[i-1,-3]) # y_sbi
        Y_tot.append(next_week_day[i-1,-4]) # y_tot
    X = np.stack(X,axis=0)
    Y = np.stack(Y,axis=0)
    Y_tot = np.stack(Y_tot,axis=0)
    return X,Y,Y_tot

def dataGenerating(data_of_days):
    # data_of_days is a list of certain day (e.g. Monday) data
    # each data is a (1440,10) ndarray
    X = []
    Y = []
    Y_tot = []
    for i in range(1,len(data_of_days)):
        # time checking
        date_cur = date(data_of_days[i-1][0][0],data_of_days[i-1][0][1],data_of_days[i-1][0][2])
        date_next = date(data_of_days[i][0][0],data_of_days[i][0][1],data_of_days[i][0][2])
        # we use current day to predict next week same day
        if (date_next - date_cur).days == 7:
            X_sub,Y_sub,Y_sub_tot = dataCombing(data_of_days[i-1],data_of_days[i])
            X.append(X_sub)
            Y.append(Y_sub)
            Y_tot.append(Y_sub_tot)
    X = np.concatenate(X,axis=0)
    Y = np.concatenate(Y,axis=0)
    Y_tot = np.concatenate(Y_tot,axis=0)
    return X,Y,Y_tot
    
#===================== Error function =====================
    
def bicycleError(Y_pred,Y,Y_tot):
    # both of them are NDarray
    return np.mean(3 * np.abs((Y - Y_pred) / Y_tot) * (np.abs(Y / Y_tot - 1/3) + np.abs(Y / Y_tot - 2/3)))

def bicycleLossFunction(beta, X, Y,Y_tot):
    return bicycleError(X@beta,Y,Y_tot)


#===================== Some stuff about submitting csv file =====================

def getTimestamp(result:pd.DataFrame):
    result["ID"] = result["id"].str[9:18]
    result["date"] = result["id"].str[0:8]
    return (result.groupby("date")["ID"].count()).index.values

def getPredictTimeStamp(result,many_days):
    Timestamp = getTimestamp(result)
    prev_Timestamp = []
    for time in Timestamp:
        while True:
            time = timeCalculator(time,7)
            if time in many_days:
                prev_Timestamp.append(time)
                break
    return prev_Timestamp

def getTrueLabel(NTU_station,time_stamp):
    mask = np.arange(0,1440,20)
    NTU_station.sort()
    Y = []
    Y_tot = []
    for sta in NTU_station:
        for time in time_stamp:
            cal = myDataloaderCertainDay(sta,time)
            if cal == "FileNotFound":
                print("FileNotFound")
                return
            Y.append(cal[mask,-3])
            Y_tot.append(cal[mask,-4])
    return np.concatenate(Y), np.concatenate(Y_tot)
        