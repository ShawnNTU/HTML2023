import numpy as np
seed = 48763
mean_pos = [3,2]
cov_pos = [[0.4,0],[0,0.4]]
mean_neg = [5,0]
cov_neg = [[0.6,0],[0,0.6]]
mean_out = [0,6]
conv_out = [[0.1,0],[0,0.3]]

def outlierProcess(size,random_seed):
    rng = np.random.default_rng(random_seed)
    y_label = np.ones(size)
    temp = []
    for _ in range(size):
        temp.append(rng.multivariate_normal(mean=mean_out,cov=conv_out,size=1))
    x_out = np.concatenate(temp,axis=0)
    x_out = np.column_stack((np.ones(size),x_out))
    return (np.matrix(y_label).reshape(-1,1),np.matrix(x_out))

def specialProcess(size,random_seed):
    rng = np.random.default_rng(random_seed)
    y_label = rng.integers(low=0,high=2,size=size)
    y_label[y_label==0] = -1
    temp = []
    for y in y_label:
        if y == 1:
            temp.append(rng.multivariate_normal(mean=mean_pos,cov=cov_pos,size=1))
        else:
            temp.append(rng.multivariate_normal(mean=mean_neg,cov=cov_neg,size=1))
            
    x_train = np.concatenate(temp,axis=0)
    x_train = np.column_stack((np.ones(size),x_train))
    return (np.matrix(y_label).reshape(-1,1),np.matrix(x_train))

TRAIN_SIZE = 256
TEST_SIZE = 4096
ITERATION = 500
ETA = 0.1
OUTLIER_SIZE = 16

def linearRegression(X:np.matrix,Y:np.matrix):
    # X : n x (d+1) np matrix
    # y : n x 1 np matrix
    pseudo_inverse = np.linalg.inv(X.T @ X) @ X.T
    w_LIN = pseudo_inverse @ Y
    return w_LIN # (d + 1) x 1

def zeroOneError(weight:np.matrix,X:np.matrix,Y:np.matrix):
    # X : n x (d + 1) np matrix
    # y : n x 1 np matrix
    # w_LEN : (d + 1) x 1 np matrix
    pred_y = np.sign(X @ weight)
    pred_y[pred_y==0] = -1
    N = Y.shape[0]
    diff = (N - np.sum(np.multiply(pred_y,Y))) / 2 / N
    return diff

def squaredError(w_LEN:np.matrix,X:np.matrix,Y:np.matrix):
    # X : n x (d + 1) np matrix
    # y : n x 1 np matrix
    # w_Len : (d + 1) x 1 np matrix
    pred_y = X @ w_LEN
    diff = np.mean(np.square(pred_y - Y))
    return diff



def logisticRegression(X:np.matrix,Y:np.matrix,iteration:int,eta:float):
    # X : n x (d + 1) np matrix
    # y : n x 1 np matrix
    N, dim = X.shape
    w_LOGREG = np.matrix(np.zeros(dim)).T # (d + 1) x 1
    for _ in range(iteration):
        neg_ytimesx = -1 * np.multiply(X,Y)
        score = 1 / (1 + np.exp(-1 * neg_ytimesx @ w_LOGREG))
        # grad = np.sum(np.multiply(score,neg_ytimesx),axis=0).reshape(-1,1)
        grad = np.mean(np.multiply(score,neg_ytimesx),axis=0).reshape(-1,1)
        w_LOGREG = w_LOGREG - eta * grad
    return w_LOGREG

def crossEntropy(weight:np.matrix,X:np.matrix,Y:np.matrix):
    pred_y = -1 * np.sum(np.log(1 / (1 + np.exp(-1 * np.multiply(X,Y) @ weight))))
    return pred_y
    