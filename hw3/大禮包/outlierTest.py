import numpy as np
import multiprocessing as mp
import myfunc
from myfunc import ETA,ITERATION,TRAIN_SIZE,TEST_SIZE,OUTLIER_SIZE

def experiment(seed):
    seed = seed
    y_train_label, x_train_data = myfunc.specialProcess(TRAIN_SIZE,random_seed=seed)
    y_outlier_label, x_outlier_data = myfunc.outlierProcess(OUTLIER_SIZE,random_seed=seed)
    y_test_label, x_test_data = myfunc.specialProcess(TEST_SIZE,random_seed=seed)

    y_outlier_label = np.concatenate((y_train_label,y_outlier_label),axis=0)
    x_outlier_data = np.concatenate((x_train_data,x_outlier_data),axis=0)
    # LIN REG
    w_LIN = myfunc.linearRegression(x_train_data,y_train_label) 
    Zero_One_LIN_out = myfunc.zeroOneError(w_LIN,x_test_data,y_test_label) # Q11
    # LOGREG
    w_LOGREG = myfunc.logisticRegression(x_train_data,y_train_label,ITERATION*10,ETA)
    Zero_One_LOGREG_out = myfunc.zeroOneError(w_LOGREG,x_test_data,y_test_label) # Q11
    
    # LIN REG with outlier
    w_LIN_outlier = myfunc.linearRegression(x_outlier_data,y_outlier_label)
    Zero_One_LIN_outlier_out = myfunc.zeroOneError(w_LIN_outlier,x_test_data,y_test_label) # Q12
    # LOGREG with outlier
    w_LOGREG_outlier = myfunc.logisticRegression(x_outlier_data,y_outlier_label,ITERATION*10,ETA)
    Zero_One_LOGREG_outlier_out = myfunc.zeroOneError(w_LOGREG_outlier,x_test_data,y_test_label) # Q12
    return (Zero_One_LIN_out,Zero_One_LOGREG_out,Zero_One_LIN_outlier_out,Zero_One_LOGREG_outlier_out)

if __name__ == "__main__":
    exp_times = 128
    pool = mp.Pool(mp.cpu_count())
    results = pool.map(experiment,range(exp_times))
    np.savetxt("outlier_result.txt",results,header="Zero_One_LIN_out,Zero_One_LOGREG_out,Zero_One_LIN_outlier_out,Zero_One_LOGREG_outlier_out")
    