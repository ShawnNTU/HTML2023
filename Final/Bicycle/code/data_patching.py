import json
import os
import copy
import multiprocessing as mp

class CustomEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, dict):
            return '{\n  ' + ', \n  '.join(f'{json.dumps(k)}:{json.dumps(v)}' for k, v in o.items()) + '\n}'
        return super().encode(o)
    
def missingFixing(path:str):
    with open(path,'r') as F:
        data = json.load(fp=F)
        keys = list(data.keys())
        length = len(keys)
        # index from -1 to -length
        prev_data = {}
        for index in range(1,length+1):
            if data[keys[-index]] == {}:
                if prev_data != {}:
                    data[keys[-index]] = copy.deepcopy(prev_data)
                    data[keys[-index]]["Loss"] = 1
                else:
                    continue
            else:
                prev_data = copy.deepcopy(data[keys[-index]])
                if "Loss" not in data[keys[-index]].keys():
                    data[keys[-index]]["Loss"] = 0
        # if corrected > 0:
    with open(path,'w') as F:
        F.writelines(json.dumps(data,cls=CustomEncoder))

# dealing with missing data that has no value from the end
# but 10/15 lost a lot of data, so I just delete data of that day
def calculateNextday(day:str):
    # extract month and date from current day
    month = int(day[-4:-2]) 
    date = int(day[-2:])
    month_31 = [1,3,5,7,8,10,12]
    if month in month_31:
        if date != 31:
            next_date = date + 1
            next_month = month
        else:
            next_date = 1
            if month != 12:
                next_month = month + 1
            else:
                next_month = 1
    else:
        if date != 30:
            next_date = date + 1
            next_month = month
        else:
            next_date = 1
            next_month = month + 1
    return next_date,next_month
        
# dealing with missing data that has no value from the end
def missingFixingTailed(day,sta):
    if sta[-5:] != ".json":
        sta = sta + ".json"
    path = f"./release/{day}/{sta}"
    corrected = 0
    with open(path,'r') as F:
        data = json.load(fp=F)
        keys = list(data.keys())
        # if list(data.values())[-1] == {}:
        if list(data.values())[-1] == {}:
            corrected = 1
            
            next_date,next_month = calculateNextday(day)
            next_path = f"./release/2023{next_month:02d}{next_date:02d}/{sta}" # data (path) of the day after current date
            
            while True: # some station doesn't have data in the next day
                try:
                    with open(next_path,'r') as next_F:
                        next_data = json.load(fp=next_F)
                        next_keys = list(next_data.keys())
                    break
                except FileNotFoundError:
                    next_date,next_month = calculateNextday(f"{next_month:02d}{next_date:02d}")
                    next_path = f"./release/2023{next_month:02d}{next_date:02d}/{sta}" # data (path) of the day after current date
                    continue
            
            data[keys[-1]] = copy.deepcopy(next_data[next_keys[0]])
            data[keys[-1]]["Loss"] = 1
            for i in range(2,1440):
                # i from -2 to -1440
                if data[keys[-i]] == {}:
                    data[keys[-i]] = data[keys[-i + 1]]
                    # here, whether they are same memory data doesn't matter
                else:
                    break
    if corrected == 1:
        with open(path,'w') as F:
            F.writelines(json.dumps(data,cls=CustomEncoder))
   
def patching(day):
    stations = os.listdir(f"./release/{day}")
    for sta in stations:
        # missingFixing(f"./release/{day}/{sta}")
        missingFixingTailed(day,sta)
    return None
        
if __name__ == "__main__":
    many_days = os.listdir('./release')
    pool = mp.Pool(mp.cpu_count())
    res = pool.map(patching,many_days)
    