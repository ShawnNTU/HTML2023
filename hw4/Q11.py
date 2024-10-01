from liblinear.liblinearutil import *
import numpy as np

data = np.genfromtxt("cooked_train")


# lam = np.power(10.0, np.array([-6, -4, -2, 0, 2]))
lam = np.power(10.0, np.array([2, 0, -2, -4, -6]))
lam = 1 / (2 * lam)

result = [0, 0, 0, 0, 0]

# for loops in range(128*5):
for loops in range(128):

    np.random.seed(loops)
    np.random.shuffle(data)
    # data = np.random.permutation(data)
    x_train = data[:, :-1]
    y_label = data[:, -1]
    prob = problem(y_label[:120], x_train[:120])
    train_model = []
    for l in lam:
        param = parameter(f"-s 0 -e 0.000001 -c {l} -q")
        train_model.append(train(prob, param))
    res = []
    for m in train_model:
        p_label, p_acc, p_val = predict(y_label[120:], x_train[120:], m, "-q")
        res.append(p_acc[0])
    result[np.argmax(res)] += 1
np.savetxt("Q11result_t", result, "%d")
# np.savetxt("Q11result2", result, "%d")
