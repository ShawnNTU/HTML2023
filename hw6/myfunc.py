import numpy as np
from io import TextIOWrapper
class Node():
    def __init__(self) -> None:
        self.left = None
        self.right = None
        self.parent = None
        
        self.best_dim = None
        self.best_theta = None
        self.best_y = None
        self.is_leaf = False
    def updateData(self,best_dim,best_s,best_theta):
        self.best_dim = best_dim
        self.best_s = best_s
        self.best_theta = best_theta
    

def check_all_same_y(y_array:np.array):
    N = y_array.shape
    y_1 = y_array[0]
    y_1_array = np.ones(N) * y_1
    # print(y_array,y_1_array)
    return np.all(y_1_array == y_array)

def check_all_same_x(x_array:np.array):
    N, dim = x_array.shape
    x_1 = x_array[0]
    x_1_array = np.ones(N*dim).reshape(N,dim) * x_1
    return np.all(x_1_array == x_array)

def decisionStumpRegression(data:np.array):
    # data : (x,y)
    N, _ = data.shape
    data = data[np.argsort(data,axis=0)[:,0]] # sort along x value
    impurity = np.zeros(N)
    for index in range(1,N):
        left_impurity = np.mean(np.square(data[:index,1] - np.mean(data[:index,1])))
        right_impurity = np.mean(np.square(data[index:,1] - np.mean(data[index:,1])))
        # left_size = index, right_size = N - index
        impurity[index] = left_impurity * index + right_impurity * (N - index)
    impurity[0] = np.sum(np.square(data[:,1] - np.mean(data[:,1])))
    best_index = int(np.median(np.arange(N)[impurity == impurity.min()]))
    # 0:best_index is left part, best_index:last is right part
    if best_index == 0:
        best_theta =  -np.inf
    else:
        best_theta = (data[best_index - 1,0] + data[best_index,0]) / 2
    return best_index,impurity[best_index],best_theta

def updateBestParameter(root:Node,data:np.array):
    N, dim = data.shape
    best_impurity = np.inf
    for d in range(dim-1,0,-1): # dim includes y, so actually dim is dim + 1
        temp_index,temp_impurity,temp_theta = decisionStumpRegression(np.column_stack((data[:,d],data[:,0]))) # (x,y)
        if temp_impurity <= best_impurity:
            best_index = temp_index
            best_impurity = temp_impurity
            best_theta = temp_theta
            best_dim = d
    
    root.best_dim = best_dim
    root.best_theta = best_theta
    root.best_y = -1

    return best_index


def updateBestParameterV2(root:Node,data:np.array):
    N, dim = data.shape
    best_impurity = np.inf
    for d in range(dim-1,0,-1): # dim includes y, so actually dim is dim + 1
        unique_value = np.unique(data[:,d])
        for i in range(len(unique_value)-1):
            theta = (unique_value[i] + unique_value[i+1])/2
            left_mask = data[:,d] < theta
            right_mask = data[:,d] > theta
            y_left = data[left_mask,0]
            y_right = data[right_mask,0] 
            
            y_left_imp = 0 if np.size(y_left) == 0 else np.mean(np.square(y_left - np.mean(y_left))) * np.size(y_left)
            y_right_imp = 0 if np.size(y_right) == 0 else np.mean(np.square(y_right - np.mean(y_right))) * np.size(y_right)
            
            impurity = y_left_imp + y_right_imp
            if best_impurity >= impurity:
                best_impurity = impurity
                best_dim = d
                best_theta = theta
    
    root.best_dim = best_dim
    root.best_theta = best_theta
    root.best_y = -1

def CARTV2(data:np.array,root:Node):
    check_y_result = check_all_same_y(data[:,0])
    if check_y_result == True:
        root.is_leaf = True
        root.best_y = data[0][0]
        return
    check_x_result = check_all_same_x(data[:,1:])
    if check_x_result == True:
        root.is_leaf = True
        root.best_y = np.mean(data[:,0])
        return
    updateBestParameterV2(root,data)
    best_theta = root.best_theta
    best_dim = root.best_dim
    left_mask = data[:,best_dim] < best_theta
    right_mask = data[:,best_dim] > best_theta
    
    y_left = data[left_mask]
    y_right = data[right_mask] 
    
    root.left = Node()
    root.left.parent = root
    
    root.right = Node()
    root.right.parent = root
    
    
    CARTV2(y_left,root.left)
    CARTV2(y_right,root.right)


def CART(data:np.array,root:Node):
    check_y_result = check_all_same_y(data[:,0])
    if check_y_result == True:
        root.is_leaf = True
        root.best_y = data[0][0]
        return
    check_x_result = check_all_same_x(data[:,1:])
    if check_x_result == True:
        root.is_leaf = True
        root.best_y = np.mean(data[:,0])
        return
    best_index = updateBestParameter(root,data)
    
    if best_index == 0:
        root.best_y = np.mean(data[:,0])
        root.is_leaf = True
    else:
        # sort along x value
        left_part = data[np.argsort(data,axis=0)[:,root.best_dim]][:best_index]
        right_part = data[np.argsort(data,axis=0)[:,root.best_dim]][best_index:]
        
        root.left = Node()
        root.left.parent = root
        
        root.right = Node()
        root.right.parent = root
        
        
        CART(left_part,root.left)
        CART(right_part,root.right)
        
        
def printTree(root:Node,level:int,file:TextIOWrapper):
    if root != None:
        printTree(root.left,level+1,file)
        print("\t" * level,end="")
        file.write("\t" * level)
        if root.is_leaf == True:
            print(f"B_y:{root.best_y}, level:{level}")
            file.write(f"B_y:{root.best_y}, level:{level}\n")
        else:
            print(f"B_dim:{root.best_dim}, B_theta:{root.best_theta}, level:{level}")
            file.write(f"B_dim: {root.best_dim}, B_theta: {root.best_theta}, level:{level}\n")
        printTree(root.right,level+1,file)
        

def CARTprediction(root:Node,data:np.array):
    if root != None:
        if root.is_leaf == True:
            data[:,-1] = root.best_y
            return data[:,-2:]
        else:
            best_dim = root.best_dim
            best_theta = root.best_theta
            
            left = data[data[:,best_dim] < best_theta]
            right = data[data[:,best_dim] >= best_theta]
            if left.size != 0 and right.size != 0: 
                y_pred_left = CARTprediction(root.left,left)
                y_pred_right = CARTprediction(root.right,right)
                return np.row_stack((y_pred_left,y_pred_right))
            elif left.size != 0:
                return CARTprediction(root.left,left)
            elif right.size != 0:
                return CARTprediction(root.right,right)

# def CARTpredictionV2(data:np.array,root=ROOT):
#     if root != None:
#         if root.is_leaf == True:
#             return root.best_y
#         else:
#             best_dim = root.best_dim
#             best_theta = root.best_theta
#             if data[best_dim] <= best_theta:
#                 return CARTpredictionV2(data,root.left)
#             else:
#                 return CARTpredictionV2(data,root.right)
