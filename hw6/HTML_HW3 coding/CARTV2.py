import libsvm.svmutil as svm
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

from myfunc import CARTV2,Node

y_train, x_train = svm.svm_read_problem("train.txt",return_scipy=True)
y_train = y_train.reshape(-1,1)
x_train = x_train.toarray()
train_data = np.concatenate((y_train,x_train),axis=1)

y_test, x_test = svm.svm_read_problem("test.txt",return_scipy=True)
y_test = y_test.reshape(-1,1)
x_test = x_test.toarray()
test_data = np.concatenate((y_test,x_test),axis=1)


N = train_data.shape[0]

def plantTreeV2(seed):
    np.random.seed(seed)
    sampling = np.random.randint(0,N,int(N/2))
    root = Node()
    CARTV2(train_data[sampling],root) 
    return root

# with open("result.txt",'w') as F:
#     printTree(root,0,F)




def CARTpredictionV2(data:np.array,root:Node):
    if root != None:
        if root.is_leaf == True:
            return root.best_y
        else:
            best_dim = root.best_dim
            best_theta = root.best_theta
            if data[best_dim] <= best_theta:
                return CARTpredictionV2(data,root.left)
            else:
                return CARTpredictionV2(data,root.right)

if __name__ == "__main__":
    root = Node()
    CARTV2(train_data,root)
    y_pred = []
    for x in test_data:
        y_pred.append(CARTpredictionV2(x,root))
    y_pred = np.array(y_pred) 
    # y_pred = np.array(pool.map(CARTpredictionV2,test_data))
    print(np.mean(np.square(y_pred - test_data[:,0])))
    #>>>8.710337768679631
    
    pool = mp.Pool(20)
    tree_list = pool.map(plantTreeV2,range(2000))
    
    pred_in = np.zeros(train_data.shape[0] * 2000).reshape(-1,2000)
    pred_out = np.zeros(test_data.shape[0] * 2000).reshape(-1,2000)
    
    for i in range(2000):
        for idx, x in enumerate(train_data):
            pred_in[idx][i] += CARTpredictionV2(x,tree_list[i])
        for idx, x in enumerate(test_data):
            pred_out[idx][i] += CARTpredictionV2(x,tree_list[i])
    
    np.savetxt("E_in.txt",pred_in,"%4f")
    np.savetxt("E_out.txt",pred_out,"%4f")
    # pred_in = np.genfromtxt("./E_in.txt")
    # pred_out = np.genfromtxt("./E_out.txt")
    G_in = np.mean(pred_in,axis=1)
    G_out = np.mean(pred_out,axis=1)
    
    MSE_in = np.mean(np.square(pred_in - train_data[:,0].reshape(-1,1)),axis=0)
    MSE_out = np.mean(np.square(pred_out - test_data[:,0].reshape(-1,1)),axis=0)
    MSE_in_G = np.mean(np.square(G_in - train_data[:,0]),axis=0)
    MSE_out_G = np.mean(np.square(G_out - test_data[:,0]),axis=0)
    
# plotting

# Q10 histogram
plt.figure(figsize=(10,5))
plt.hist(MSE_out,bins=np.arange(7,12,0.5))
plt.xticks(np.arange(7,12,0.5))
plt.yticks(np.arange(100,1100,100))
plt.grid()
plt.savefig("Q10_plot")
plt.close()

# Q11 scatter plot

plt.figure(figsize=(10,5))
plt.scatter(MSE_in,MSE_out)
plt.scatter([MSE_in_G],[MSE_out_G],s=20,c="red")
plt.xticks(np.arange(0,12))
plt.xlabel("$E_{in}$ MSE")
plt.yticks(np.arange(0,12))
plt.ylabel("$E_{out}$ MSE")
plt.grid()
plt.savefig("Q11_plot")
plt.close()


avg_err = []

for i in range(1,2001):
    G_out_t = np.mean(pred_out[:,:i],axis=1)
    MSE_out_G_t = np.mean(np.square(G_out_t - test_data[:,0]),axis=0)
    avg_err.append(MSE_out_G_t)
    
plt.figure(figsize=(15,5))
plt.plot(avg_err)
plt.plot(MSE_out)
plt.yticks(np.arange(4,12,0.5))
plt.ylabel("$E_{out}(G_{t})$ MSE")
plt.xticks(np.arange(0,2100,100))
plt.tight_layout()
plt.savefig("Q12_plot")
plt.close()
