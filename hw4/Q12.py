from liblinear.liblinearutil import *
import numpy as np


data = np.genfromtxt("cooked_train")

lam = np.power(10.0, np.array([2, 0, -2, -4, -6]))
lam = 1 / (2 * lam)

result = [0, 0, 0, 0, 0]

# for loops in range(128):

#     np.random.seed(loops)
#     np.random.shuffle(data)
#     x_train = data[:, :-1]
#     y_label = data[:, -1]
#     prob = problem(y_label, x_train)
#     res = []
#     for l in lam:
#         param = parameter(f"-s 0 -e 0.000001 -c {l} -v 5 -q")
#         res.append(train(prob, param))
#     result[np.argmax(res)] += 1

CAT = np.concatenate

for loops in range(128):

    np.random.seed(loops)
    np.random.shuffle(data)
    five_fold = np.split(data,indices_or_sections=5)
    fold0 = CAT((five_fold[1],five_fold[2],five_fold[3],five_fold[4]),axis=0)
    fold1 = CAT((five_fold[0],five_fold[2],five_fold[3],five_fold[4]),axis=0)
    fold2 = CAT((five_fold[0],five_fold[1],five_fold[3],five_fold[4]),axis=0)
    fold3 = CAT((five_fold[0],five_fold[1],five_fold[2],five_fold[4]),axis=0)
    fold4 = CAT((five_fold[0],five_fold[1],five_fold[2],five_fold[3]),axis=0)
    
    prob0 = problem(fold0[:,-1], fold0[:,:-1])
    prob1 = problem(fold1[:,-1], fold1[:,:-1])
    prob2 = problem(fold2[:,-1], fold2[:,:-1])
    prob3 = problem(fold3[:,-1], fold3[:,:-1])
    prob4 = problem(fold4[:,-1], fold4[:,:-1])
    
    # validate = [five_fold[0],five_fold[1],five_fold[2],five_fold[3],five_fold[4]]
    problems = [prob0,prob1,prob2,prob3,prob4]

    res = []
    for l in lam:
        train_model = []
        for index in range(5):
            param = parameter(f"-s 0 -e 0.000001 -c {l} -q")
            train_model.append(train(problems[index], param))
        acc = 0
        for index in range(5):
            p_label, p_acc, p_val = predict(five_fold[index][:,-1], five_fold[index][:,:-1], train_model[index],"-q")
            acc += p_acc[0]
        res.append(acc/5)
    result[np.argmax(res)] += 1
np.savetxt("Q12result", result, "%d")
