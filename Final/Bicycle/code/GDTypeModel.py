import numpy as np
import pandas as pd
from scipy.optimize import minimize


from myfunction import dataGenerating, bicycleLossFunction, dateConversion
from myfunction import myDataloader, myDataloaderCertainDay
from myfunction import dataCombing, bicycleError

import datetime

class GDTypeModel:
    def __init__(self) -> None:
        self.train_start_date = None
        self.train_end_date = None
        self.GD_model = None
        
    def prediction(self, cal_previous_week, day, hour:int, minute:int):
        data = cal_previous_week[day][0][:,-3]
        index = hour * 60 + minute
        X = data[index-20:index]
        return self.GD_model[day].x @ X

    
    def training(self, seven_days_data, train_interval):
        
        self.train_start_date = train_interval[0]
        self.train_end_date = train_interval[-1]
        
        model_7days = []
        for one_day_data in seven_days_data:
            X_train, Y_train, Y_train_tot = dataGenerating(one_day_data)
            weight_init = np.random.randint(low=1,high=Y_train_tot.max(),size=X_train.shape[1])
            model_7days.append(minimize(bicycleLossFunction,weight_init,args=(X_train,Y_train,Y_train_tot),method="L-BFGS-B"))
            # model_7days.append(minimize(bicycleLossFunction,weight_init,args=(X_train,Y_train,Y_train_tot),method="Powell"))
        self.GD_model = model_7days
            
def generateGDModels(NTU_station,train_interval):
    NTU_GD_model = {}
    for sta in NTU_station:
        NTU_GD_model[sta] = GDTypeModel()
        cal = myDataloader(sta,train_interval)
        NTU_GD_model[sta].training(cal,train_interval)
    return NTU_GD_model
    
    
def GDpredictionDataPreparing():
    
    pass

def predictCSVWithGDodel(df:pd.DataFrame, NTU_model):
    for i in df.index:
        id = df.loc[i,"id"]
        sta = id[9:18]
        # deal with time
        day,hour,minute = dateConversion(id)
        dateinfo = datetime.date(int(id[:4]),int(id[4:6]),int(id[6:8]))
        while True:
            dateinfo = dateinfo - datetime.timedelta(7)    
            cal = myDataloader(sta,[f"{dateinfo.year:04d}{dateinfo.month:02d}{dateinfo.day:02d}"])
            if cal != "FileNotFound":
                break
        # if we want predict 00:00, I just make it same as 00:20 prediction
        if hour == 0 and minute == 0:
            df.loc[i,"sbi"] = NTU_model[sta].prediction(cal,day, hour, minute+20)
        else:
            df.loc[i,"sbi"] = NTU_model[sta].prediction(cal,day, hour, minute)
    return df

def optimizePrediction(NTU_model, NTU_station, prev_interval, next_interval):
    NTU_station.sort()
    prediction = []
    for sta in NTU_station:
        res = []
        for i in range(7):
            cal_prev = myDataloaderCertainDay(sta,prev_interval[i])
            cal_next = myDataloaderCertainDay(sta,next_interval[i])
            
            X,Y,Y_tot= dataCombing(cal_prev,cal_next)
            Y_pred = X @ NTU_model[sta].GD_model[cal_prev[0][3]].x
            # Y_pred above is the prediction of 00:19, 00:20,..., 23:59
            
            # we let 00:00~00:18 prediction same as 00:19
            Y = np.concatenate((np.tile([Y[0]],19),Y))
            Y_tot = np.concatenate((np.tile([Y_tot[0]],19),Y_tot))
            Y_pred = np.concatenate((np.tile([Y_pred[0]],19),Y_pred))
            res.append(bicycleError(Y_pred,Y,Y_tot))
        prediction.append(np.mean(res))
    return prediction
        
            
