import numpy as np
import libsvm.svmutil as svm
import matplotlib.pyplot as plt
import random

y_train, x_train = svm.svm_read_problem("train.txt")
y_test, x_test = svm.svm_read_problem("test.txt")

# ========== 9 ~ 11 ==========
# ========== 9 ==========
# y_train_9 = np.where(np.array(y_train) == 4, 1, -1)

# cost_arr = [0.1,1,10]
# degree_arr = [2,3,4]

# for c in cost_arr:
#     for degree in degree_arr:
#         m = svm.svm_train(y_train_9,x_train,f"-s 0 -t 1 -d {degree} -g 1 -r 1 -c {c} -q")
#         print(f"({c},{degree}):{m.get_nr_sv()}")

# (0.1,2):860
# (0.1,3):789
# (0.1,4):740
# (1,2):783
# (1,3):721
# (1,4):666
# (10,2):712
# (10,3):659
# (10,4):629

# ========== 10 ==========
# y_train_10 = np.where(np.array(y_train) == 1, 1, -1)
# y_test_10 = np.where(np.array(y_test) == 1, 1, -1)

# cost_arr = [0.01,0.1,1,10,100]

# for c in cost_arr:
#     m = svm.svm_train(y_train_10,x_train,f"-s 0 -t 2 -g 1 -c {c} -q")
#     _,acc,_ = svm.svm_predict(y_test_10,x_test,m)
#     print(f"{c}:{acc[0]}")

# Accuracy = 95.4% (1908/2000) (classification)
# 0.01:95.39999999999999
# Accuracy = 98.8% (1976/2000) (classification)
# 0.1:98.8
# Accuracy = 99.5% (1990/2000) (classification)
# 1:99.5
# Accuracy = 99.4% (1988/2000) (classification)
# 10:99.4
# Accuracy = 99.45% (1989/2000) (classification)
# 100:99.45

# ========== 11 ==========
y_train_11 = np.where(np.array(y_train) == 1, 1, -1)
y_test_11 = np.where(np.array(y_test) == 1, 1, -1)

cost_arr = [0.01,0.1,1,10,100]
cost_count = [0,0,0,0,0]
combined = list(zip(y_train_11,x_train))

record = []
for seed in range(1000):
    np.random.seed(seed)
    np.random.shuffle(combined) # random and np.random will lead to different result
    y_part, x_part = list(zip(*combined))
    y_valid_11 = y_part[:200]
    x_valid_11 = x_part[:200]
    y_train_11 = y_part[200:]
    x_train_11 = x_part[200:]
    
    
    temp = []
    for c in cost_arr:
        m = svm.svm_train(y_train_11,x_train_11,f"-s 0 -t 2 -g 1 -c {c} -q")
        _,acc,_ = svm.svm_predict(y_valid_11,x_valid_11,m,"-q")
        temp.append(acc[0])
    cost_count[np.argmax(temp)] += 1
    record.append(temp)
    print(f"Current loop:{seed}",end="\r")
print(cost_count)
np.savetxt("record2.txt",record)
# >>> [3, 196, 396, 400, 5]
# cost_label = ["0.01","0.1","1","10","100"]
# value_label = [3, 196, 396, 400, 5]
# cost_value = [f"{cost_label[i]}\n{value_label[i]}" for i in range(5)]

# plt.figure(figsize=(10.8,12.8))
# plt.bar(cost_value, value_label)
# plt.title("Q11 Best C by validation",fontsize=20)
# plt.yticks(range(0,451,25),fontsize=20)
# plt.xticks(fontsize=20)
# plt.xlabel("Cost\ncounts",fontsize=20)
# plt.grid(axis='y')
# plt.savefig("Q11_bar_plot")
# plt.show()
# plt.close()

# # ========== 12 ==========
# y_train, x_train = svm.svm_read_problem("train.txt",return_scipy=True)
# y_train_12 = np.where(y_train == 3, 1, 0)
# x_train = x_train.toarray()
# cost_arr = [0.01,0.1,1,10,100]

# # \sum \sum  a_n*a_m*y_n*y_m K(x_n,x_m)
# w_length = []
# for c in cost_arr:
#     m = svm.svm_train(y_train_12,x_train,f"-s 0 -t 2 -g 1 -c {c} -q")
#     SV = x_train[np.array(m.get_sv_indices()) - 1]
#     SV_coef = np.array(m.get_sv_coef())
#     alpha_matrix = SV_coef * SV_coef.T
#     kernel = np.exp(-1 * np.sum(np.square(SV[:,np.newaxis,:] - SV[np.newaxis,:,:]),axis=2))
#     w_length.append(np.sqrt(np.sum(alpha_matrix * kernel)))
    
# print(w_length)
#>>> [2.7047263578305563, 5.109372234199461, 12.990758744184518, 37.36384509923585, 84.48721707631326]

# cost = [0.01,0.1,1,10,100]
# w_length = [2.7047263578305563, 5.109372234199461, 12.990758744184518, 37.36384509923585, 84.48721707631326]

# plt.figure(figsize=(10.8,10.8))
# plt.plot(cost, w_length)
# plt.scatter(cost, w_length,s=20,c="red")
# plt.title("Q12 Length of $\mathbf{w}$",fontsize=20)
# plt.xscale(value='log')
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# plt.xlabel("C",fontsize=20)
# plt.savefig("Q12_bar_plot")
# plt.show()
# plt.close()