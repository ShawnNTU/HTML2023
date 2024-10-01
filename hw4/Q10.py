from liblinear.liblinearutil import *
import numpy as np

lam = np.power(10.0, np.array([-6, -4, -2, 0, 2]))
lam = 1 / (2 * lam)

data = np.genfromtxt("cooked_train")
x_train = data[:, :-1]
y_label = data[:, -1]

prob = problem(y_label, x_train)

train_model = []
for l in lam:
    param = parameter(f"-s 0 -e 0.000001 -c {l} -q")
    train_model.append(train(prob, param))

result = []
for m in train_model:
    p_label, p_acc, p_val = predict(y_label, x_train, m)
    result.append(p_acc[0])

for r in result:
    print(f"{r:.3f}%")

# 96.000% -6
# 92.000% -4
# 91.000% -2
# 87.500%  0
# 80.500%  2
