import numpy as np


def transformer(x):
    one_order = list(x)
    second_order = list(np.power(x, 2))
    third_order = list(np.power(x, 3))
    second_order_combine = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            second_order_combine.append(x[i]*x[j])
    third_order_combine_square = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            third_order_combine_square.append(x[i]*x[i]*x[j])
            third_order_combine_square.append(x[i]*x[j]*x[j])
    third_order_combine = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for k in range(j+1, len(x)):
                third_order_combine.append(x[i]*x[j]*x[k])
    return [1]+one_order+second_order+third_order+second_order_combine+third_order_combine_square+third_order_combine


if __name__ == "__main__":
    raw_data = np.genfromtxt("train.txt")
    x_raw = raw_data[:, :-1]
    y_label = raw_data[:, -1].reshape(-1, 1)
    cooked_x = []
    for x in x_raw:
        cooked_x.append(transformer(x))
    np.savetxt("cooked_train", np.concatenate((cooked_x, y_label), axis=1),"%.4f")

# # ========================================
# raw_data = np.genfromtxt("train.txt")
# x_raw = raw_data[:, :-1]
# y_label = raw_data[:, -1]
# cooked_x = []
# for x in x_raw:
#     cooked_x.append(transformer(x))
# with open("cooked_train.txt", 'w') as F:
#     for i in range(y_label.shape[0]):
#         if int(y_label[i]) == 1:
#             F.write(f"+{int(y_label[i])} ")
#         else:
#             F.write(f"{int(y_label[i])} ")
#         for index, x in enumerate(cooked_x[i], 1):
#             F.write(f"{index}:{x:.6f} ")
#         F.write("\n")
# # ========================================
# raw_data = np.genfromtxt("test.txt")
# x_raw = raw_data[:, :-1]
# y_label = raw_data[:, -1]
# cooked_x = []
# for x in x_raw:
#     cooked_x.append(transformer(x))
# with open("cooked_test.txt", 'w') as F:
#     for i in range(y_label.shape[0]):
#         if int(y_label[i]) == 1:
#             F.write(f"+{int(y_label[i])} ")
#         else:
#             F.write(f"{int(y_label[i])} ")
#         for index, x in enumerate(cooked_x[i], 1):
#             F.write(f"{index}:{x:.6f} ")
#         F.write("\n")
