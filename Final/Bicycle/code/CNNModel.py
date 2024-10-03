import torch
import numpy as np
import pandas as pd

import myfunction

class myCNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv1d_1 = torch.nn.Conv1d(in_channels=1,out_channels=32,kernel_size=20,stride=1) #  1 x 1440 -> 32 x 1421
        self.Conv1d_2 = torch.nn.Conv1d(in_channels=32,out_channels=64,kernel_size=20,stride=1) #  32 x 1421 -> 64 x 1402
        self.AvgP_1 = torch.nn.AvgPool1d(kernel_size=20,stride=20) # 64 x 1402 -> 64 x 70
        self.flat = torch.nn.Flatten(start_dim=1,end_dim=-1)
        self.Drop1 = torch.nn.Dropout(p=0.5)
        self.dense = torch.nn.Linear(in_features=64*70,out_features=1)
        self.ReLU = torch.nn.ReLU()
    def forward(self,x):
        x = self.Conv1d_1(x)
        # print(x.shape)
        x = self.ReLU(x)
        x = self.Conv1d_2(x)
        # print(x.shape)
        x = self.ReLU(x)
        x = self.AvgP_1(x)
        # print(x.shape)
        x = self.flat(x)
        # print(x.shape)
        x = self.Drop1(x)
        x = self.dense(x)
        # print(x.shape)
        x = self.ReLU(x)
        return x

def generateCNNModels(NTU_station,train_interval):
    NTU_CNN_model = {}
    for sta in NTU_station:
        NTU_CNN_model[sta] = CNNModel()
        x_train, y_train, y_tot_train = getCNNData(sta, train_interval)
        NTU_CNN_model[sta].training(x_train, y_train, y_tot_train)
    return NTU_CNN_model

def CNNModelPrediction(NTU_station, NTU_CNN_model, test_interval):
    loss = []
    for sta in NTU_station:
        x_test, y_test, y_tot_test = getCNNData(sta, test_interval)
        loss.append(NTU_CNN_model[sta].predicting(x_test, y_test, y_tot_test))
    return loss

def getCNNData(sta, interval):
    cal = []
    for day in interval:
        cal.append(myfunction.myDataloaderCertainDay(sta, day))
    cal = np.concatenate(cal,axis=0)
    size = cal.shape[0]
    x = []
    y = []
    y_tot= []
    for i in range(1440,size-1440+1,1):
        x.append(cal[i-1440:i,-3])# i-1 is endpoint
        y.append(cal[i-1+1440,-3])# +1440 to get next week
        y_tot.append(cal[i-1+1440,-4])
    x = torch.tensor(np.stack(x,axis=0)).to(torch.float32)
    y = torch.tensor(np.stack(y,axis=0)).to(torch.float32)
    y_tot = torch.tensor(np.stack(y_tot,axis=0)).to(torch.float32)
    return x,y,y_tot

def getCNNDataWeighted(sta, interval):
    cal = []
    for day in interval:
        cal.append(myfunction.myDataloaderCertainDay(sta, day))
    cal = np.concatenate(cal,axis=0)
    size = cal.shape[0]
    x = []
    y = []
    y_tot= []
    weights = []
    for i in range(1440,size-1440+1,1):
        ration = cal[i-1+1440,-3] / cal[i-1+1440,-4]
        weight = 3*(np.abs(ration-1/3)+np.abs(ration-2/3))
        x.append(np.tile(cal[i-1440:i,-3],reps=(int(weight),1)))# i-1 is endpoint
        y.append(np.tile(cal[i-1+1440,-3],reps=(int(weight),1)))# +1440 to get next week
        y_tot.append(np.tile(cal[i-1+1440,-4],reps=(int(weight),1)))
        weights.append(weight)
    x = torch.tensor(np.concatenate(x)).to(torch.float32)
    y = torch.tensor(np.concatenate(y)).to(torch.float32)
    y_tot = torch.tensor(np.concatenate(y_tot)).to(torch.float32)
    return x,y,y_tot,weights

def generateWeightedCNNModels(NTU_station,train_interval):
    NTU_CNN_model = {}
    for sta in NTU_station:
        NTU_CNN_model[sta] = CNNModel()
        x_train, y_train, y_tot_train, _ = getCNNDataWeighted(sta, train_interval)
        NTU_CNN_model[sta].training(x_train, y_train, y_tot_train)
    return NTU_CNN_model

def bicycleCriteria(labels,x,tot):
    u = (torch.abs(labels / tot - 1/3) + torch.abs(labels / tot - 2/3))
    # print(u.shape)
    diff = torch.abs((labels - x) / tot) 
    # print(diff.shape)
    return torch.mean(3 * diff * u)


class CNNModel():
    
    def __init__(self) -> None:
        self.train_start_date = None
        self.train_end_date = None
        
        self.CNN_model = myCNN()
        self.batch_size = 64
        self.device = 'cuda'
        
        self.train_loss = None
    
    
    def toDevice(self,device):
        self.CNN_model.to(device)
    
    def training(self, x_train, y_train, y_tot_train):
        device = 'cuda'
        self.toDevice(device)
        self.CNN_model.train()
        optimizer = torch.optim.Adam(self.CNN_model.parameters(),lr=1.126E-4,weight_decay=0.00001126)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=4,gamma=0.1)
        batch_size = self.batch_size
        
        train_pattern = []
        size = x_train.size()[0]
        
        for epoch in range(10):
            train_losses = []
            for i in range(0,size,batch_size):
                x = x_train[i:i+batch_size].reshape(-1,1,1440).to(device)
                y = y_train[i:i+batch_size].reshape(-1,1).to(device)
                y_tot = y_tot_train[i:i+batch_size].reshape(-1,1).to(device)
                optimizer.zero_grad()
                outputs = self.CNN_model(x)
                # print(outputs)
                loss = bicycleCriteria(y,outputs,y_tot)
                # print(loss)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            scheduler.step()
            train_pattern.append(np.mean(train_losses))
            
        self.train_loss = train_pattern
        
    def trainingFullBatch(self, x_train, y_train, y_tot_train):
        device = 'cuda'
        self.toDevice(device)
        self.CNN_model.train()
        optimizer = torch.optim.Adam(self.CNN_model.parameters(),lr=1.126E-4,weight_decay=0.00001126)
        
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=4,gamma=0.1)
        
        train_pattern = []
        for epoch in range(10):
            
            x = x_train.reshape(-1,1,1440).to(device)
            y = y_train.reshape(-1,1).to(device)
            y_tot = y_tot_train.reshape(-1,1).to(device)
            optimizer.zero_grad()
            outputs = self.CNN_model(x)
            # print(outputs)
            loss = bicycleCriteria(y,outputs,y_tot)
            # print(loss)
            loss.backward()
            optimizer.step()
            train_pattern.append(loss.item())
            # scheduler.step()
            
        self.train_loss = train_pattern
                
    def predicting(self, x_test, y_test, y_tot_test):
        batch_size = self.batch_size
        device = 'cuda'
        self.toDevice(device)
        
        size = x_test.size()[0]
        
        with torch.no_grad():
            test_losses = []
            for i in range(0,size,batch_size):
                x = x_test[i:i+batch_size].reshape(-1,1,1440).to(device)
                y = y_test[i:i+batch_size].reshape(-1,1).to(device)
                y_tot = y_tot_test[i:i+batch_size].reshape(-1,1).to(device)
                outputs = self.CNN_model(x)
                test_losses.append(bicycleCriteria(y,outputs,y_tot).item())
                
        return np.mean(test_losses)