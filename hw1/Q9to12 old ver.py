import numpy as np
import multiprocessing as mp
import time
#=========== data ===========
raw = np.genfromtxt("hw1_data.txt")
N,dim = raw.shape
cooked = np.column_stack((np.ones(N),raw))
train_y = cooked[:,-1]
raw_x = cooked[:,:-1]
proccessor_num = 24
#=========== scaling ===========
# train_x = raw_x # Q9
# train_x = raw_x * 11.26 # Scaling Q10
# train_x = raw_x # Q11
# train_x[:,0] = train_x[:,0] * 11.26 # Q11
train_x = raw_x # Q12

def PLA(seed):
    weight = np.zeros(dim)
    correct = 0
    update_num = 0
    np.random.seed(seed)
    while True:
        index = np.random.randint(low=0,high=N)
        y_pred = np.sign(weight.T @ train_x[index])
        y_pred = 1 if y_pred == 0 else y_pred
        if y_pred != train_y[index]:
            weight += train_y[index] * train_x[index]
            update_num += 1
            correct = 0
        else:
            correct += 1
        if correct == 5*N:
            break
    temp_result = np.sign(weight @ train_x.T)
    err = (N - np.sum(temp_result*train_y))/(2*N)
    return [update_num,err]

def modifiedPLA(seed):
    weight = np.zeros(dim)
    correct = 0
    update_num = 0
    np.random.seed(seed)
    while True:
        index = np.random.randint(low=0,high=N)
        y_pred = np.sign(weight.T @ train_x[index])
        y_pred = 1 if y_pred == 0 else y_pred
        if y_pred != train_y[index]:
            while True:
                weight += train_y[index] * train_x[index]
                y_sub_pred = np.sign(weight.T @ train_x[index])
                y_sub_pred = 1 if y_sub_pred == 0 else y_sub_pred
                update_num += 1
                if y_sub_pred == train_y[index]:
                    correct = 0
                    break
        else:
            correct += 1
        if correct == 5*N:
            break
    temp_result = np.sign(weight @ train_x.T)
    err = (N - np.sum(temp_result*train_y))/(2*N)
    return [update_num,err]

if __name__ == "__main__":
    # outputfile = "Q9.txt"
    # outputfile = "Q10.txt"
    # outputfile = "Q11.txt"
    outputfile = "Q12_1.txt"
    #=========== start ! ===========
    start_time = time.time()
    pool = mp.Pool(processes=proccessor_num)
    # result = pool.map(PLA,range(1000))
    result = pool.map(modifiedPLA,range(1000))
    np.savetxt(outputfile,np.array(result))
    #=========== end ! ===========
    end_time = time.time()
    print(f"{end_time - start_time : .3f} seconds")



