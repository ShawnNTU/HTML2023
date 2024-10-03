import numpy as np
import pandas as pd


from myfunction import dateConversion, myDataloader, myDataloaderCertainDay, bicycleError

class MeanValueModel:
    def __init__(self) -> None:
        self.train_start_date = None
        self.train_end_date = None
        self.mean_value = None
        self.mean_value_count = None
        
    def prediction(self, day, hour:int, minute:int):
        pred = self.mean_value[day]
        index = hour*60 + minute
        return pred[index]

    def training(self, seven_days_data, train_interval):
        self.train_start_date = train_interval[0]
        self.train_end_date = train_interval[-1]
        
        self.mean_value_count = np.zeros(7).astype(int)
        self.mean_value = []
        for i in range(7):
            days_data = seven_days_data[i]
            self.mean_value_count[i] += len(seven_days_data[i])
            self.mean_value.append(np.mean(days_data[:,:,-3],axis=0))
            
    def trainingWeighted(self, seven_days_data, train_interval):
        self.train_start_date = train_interval[0]
        self.train_end_date = train_interval[-1]
        
        self.mean_value_count = np.zeros(7).astype(int)
        self.mean_value = []
        for i in range(7):
            days_data = seven_days_data[i]
            self.mean_value_count[i] += len(seven_days_data[i])
            
            ratio = days_data[:,:,-3] / days_data[:,:,-4]
            weights = 3 * (np.abs(ratio-1/3)+np.abs(ratio-2/3))
            self.mean_value.append(np.average(days_data[:,:,-3],axis=0,weights=weights))
            
    def livelongTraining(self, days_data):
        # days_data is a list of days data
        # each days data is a (1440, 10) NDarray like seven_days_data structure
        for data in days_data:
            day = data[0][3] # extract "day" from data
            old_data = self.mean_value[day]
            new_data = data[:,-3]
            old_count = self.mean_value_count[day]
            self.mean_value[day] = old_data * (old_count/(old_count+1)) + new_data * (1/(old_count+1))
            self.mean_value_count[day] += 1


def MVprediction(NTU_station, NTU_MV_model, interval):
    prediction_loss = []
    for sta in NTU_station:
        model = NTU_MV_model[sta]
        temp = []
        for i in range(7):
            pred = model.mean_value[(i+1)%7]
            true_data = myDataloaderCertainDay(sta, interval[i])
            Y = true_data[:,-3]
            Y_tot = true_data[:,-4]
            temp.append(bicycleError(pred,Y,Y_tot))
        prediction_loss.append(np.mean(temp))
    return prediction_loss
     
def MVpredictionCertainPeriod(NTU_station, NTU_MV_model, interval, start, end):
    prediction_loss = []
    for sta in NTU_station:
        model = NTU_MV_model[sta]
        temp = []
        for i in range(7):
            pred = model.mean_value[(i+1)%7][start:end]
            true_data = myDataloaderCertainDay(sta, interval[i])
            Y = true_data[start:end,-3]
            Y_tot = true_data[start:end,-4]
            temp.append(bicycleError(pred,Y,Y_tot))
        prediction_loss.append(np.mean(temp))
    return prediction_loss       

    
def predictCSVWithMVModel(df:pd.DataFrame, NTU_model):
    for i in df.index:
        id = df.loc[i,"id"]
        sta = id[9:18]
        day,hour,minute = dateConversion(id)
        df.loc[i,"sbi"] = NTU_model[sta].prediction(day, hour, minute)
    return df

def generateMeanValueModels(NTU_station,train_interval):
    NTU_MVM_model = {}
    for sta in NTU_station:
        NTU_MVM_model[sta] = MeanValueModel()
        cal = myDataloader(sta,train_interval)
        NTU_MVM_model[sta].training(cal,train_interval)
    return NTU_MVM_model

def generateMeanValueModelsWeighted(NTU_station,train_interval):
    NTU_MVM_model = {}
    for sta in NTU_station:
        NTU_MVM_model[sta] = MeanValueModel()
        cal = myDataloader(sta,train_interval)
        NTU_MVM_model[sta].trainingWeighted(cal,train_interval)
    return NTU_MVM_model


# JUST USE PICKLE INSTEAD

# def saveMeanValueModel(NTU_Mean_Value_Model, NTU_station):
#     files = os.listdir()
#     if "MeanValueModel" not in files:
#         os.mkdir("./MeanValueModel")
#         for sta in NTU_station:
#             os.mkdir(f"./MeanValueModel/{sta}")
#     for sta in NTU_station:
#         for i in range(7):
#             np.save(f"./MeanValueModel/{sta}/mean_value_{i}",NTU_Mean_Value_Model[sta].mean_value[i])
#         np.savetxt(f"./MeanValueModel/{sta}/mean_value_counts",NTU_Mean_Value_Model[sta].mean_value_count,fmt="%d")
#         with open(f"./MeanValueModel/{sta}/dateinfo.txt",'w') as F:
#             F.write(f"train_start_date,train_end_date\n")
#             F.write(f"{NTU_Mean_Value_Model[sta].train_start_date},{NTU_Mean_Value_Model[sta].train_end_date}")
            
# def loadMeanValueModel(NTU_station):
#     files = os.listdir()
#     if "MeanValueModel" not in files:
#         print("MeanValueModel files are not in current directory!")
#         return
#     NTU_Mean_Value_Model = {}
#     for sta in NTU_station:
#         NTU_Mean_Value_Model[sta] = MeanValueModel()
#         temp = []
#         for i in range(7):
#             temp.append(np.load(f"./MeanValueModel/{sta}/mean_value_{i}.npy"))
#         NTU_Mean_Value_Model[sta].mean_value = temp
#         NTU_Mean_Value_Model[sta].mean_value_count = np.genfromtxt(f"./MeanValueModel/{sta}/mean_value_counts")
#         with open(f"./MeanValueModel/{sta}/dateinfo.txt",'r') as F:
#             F.readline()
#             dateinfo = F.readline().split(",")
#             NTU_Mean_Value_Model[sta].train_start_date = dateinfo[0]
#             NTU_Mean_Value_Model[sta].train_end_date = dateinfo[1]
#     return NTU_Mean_Value_Model