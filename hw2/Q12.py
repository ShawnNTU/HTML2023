import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

def targetFunction(x:float):
    result = -1 if x <= 0 else 1
    noise = np.random.randint(low = 1,high = 11)
    return result * -1 if noise == 1 else result

def randomDecisionStump1D(x,y):
    """_summary_
    Args:
        x (np.array): one dimension array
        y : labels array
    return:
        best theta, sign and correct number
    """
    comb = np.column_stack((x,y))
    comb = np.array(sorted(comb,key=lambda x : x[0]))
    x = comb[:,0]
    y = comb[:,1]
    s = np.random.randint(0,2)
    best_sign = -1 if s == 0 else 1
    best_theta = np.random.uniform(-1,1)
    best_correct = 0
    for index,label in enumerate(y):
        y_pred = -1 * best_sign if x[index] <= best_theta else best_sign
        if y_pred == label:
            best_correct += 1
    return best_theta,best_sign,best_correct
        
def Q12(seed):
    np.random.seed(seed)
    x_train = np.random.uniform(low= -1, high= 1,size = 8)
    y_label = list(map(lambda x : targetFunction(x),x_train))
    best_theta,best_sign,best_correct = randomDecisionStump1D(x_train,y_label)
    E_in = 1 - best_correct / len(y_label)
    E_out = 0.5 - 0.4 * best_sign + 0.4 * best_sign * np.abs(best_theta)
    return [E_in,E_out]


proccessor_num = 20

if __name__ == "__main__":
    pool = mp.Pool(processes=proccessor_num)
    result = np.array(pool.map(Q12,range(2000)))
    plt.scatter(result[:,0],result[:,1])
    plt.title(f"median:{np.median(result[:,1] - result[:,0])}",fontsize=18)
    plt.xlabel("E_in",fontsize=14)
    plt.ylabel("E_out",fontsize=14)
    plt.savefig("Q12")

        
            
        
    
    
    