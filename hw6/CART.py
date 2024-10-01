import libsvm.svmutil as svm
import numpy as np

import multiprocessing as mp

from myfunc import CART, Node, printTree, CARTprediction
from myfunc import CARTV2

y_train, x_train = svm.svm_read_problem("train.txt",return_scipy=True)
y_train = y_train.reshape(-1,1)
x_train = x_train.toarray()
train_data = np.concatenate((y_train,x_train),axis=1)

y_test, x_test = svm.svm_read_problem("test.txt",return_scipy=True)
y_test = y_test.reshape(-1,1)
x_test = x_test.toarray()
test_data = np.concatenate((y_test,x_test),axis=1)
# size = test_data.shape[0]
# test_data = np.column_stack((test_data,np.arange(size),np.zeros(size)))

N = train_data.shape[0]

def plantTree(seed):
    np.random.seed(seed)
    sampling = np.random.randint(0,N,int(N/2))
    root = Node()
    CART(train_data[sampling],root) 
    return root
# with open("result.txt",'w') as F:
#     printTree(root,0,F)


ROOT = Node()

def CARTpredictionV2(data:np.array,root=ROOT):
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
    pool = mp.Pool(20)
    
    
    
    CARTV2(train_data,ROOT)
    y_pred = []
    for x in test_data:
        y_pred.append(CARTpredictionV2(x))
    
    y_pred = np.array(y_pred) 
    # y_pred = np.array(pool.map(CARTpredictionV2,test_data))
    print(np.mean(np.square(y_pred - test_data[:,0])))
    
    # CART(train_data,root) 
    # y_pred = CARTprediction(root,test_data)
    # y_pred = y_pred[np.argsort(y_pred,axis=0)[:,0]]
    # print(np.mean(np.square(y_pred[:,1] - test_data[:,0])))
    # >>>9.654042988741043
    # result = pool.map(plantTree,range(2000))
    
    # err_in = []
    # err_out = []
    # train_data_for_pred = np.column_stack((train_data,np.arange(N),np.zeros(N)))
    # for i in range(2000):
    #     y_pred = CARTprediction(result[i],test_data)
    #     y_pred = y_pred[np.argsort(y_pred,axis=0)[:,0]]
    #     err_out.append(np.mean(np.square(y_pred[:,1] - test_data[:,0])))
        
    #     y_pred = CARTprediction(result[i],train_data_for_pred)
    #     y_pred = y_pred[np.argsort(y_pred,axis=0)[:,0]]
    #     err_in.append(np.mean(np.square(y_pred[:,1] - train_data_for_pred[:,0])))
    # np.savetxt("gt_err_out.txt",err_out,"%4f")
    # np.savetxt("gt_err_in.txt",err_in,"%4f")
    # avg_err = []
    # for i in range(1,2000+1):
    #     avg_err.append(np.mean(err[:i]))
    # np.savetxt("avg_err.txt",avg_err,"%4f")

