import numpy as np
import matplotlib.pyplot as plt

data11 = np.genfromtxt("Q11result")
# data11_2 = np.genfromtxt("Q11result2")
data12 = np.genfromtxt("Q12result")

labels = ["2", "0", "-2", "-4", "-6"]
label11 = [f"{labels[i]}\n{int(data11[i])}" for i in range(5)]
# label11_2 = [f"{labels[i]}\n{int(data11_2[i])}" for i in range(5)]
label12 = [f"{labels[i]}\n{int(data12[i])}" for i in range(5)]


# plt.figure(figsize=(10.8,10.8))
# plt.bar(label11_2, data11_2)
# plt.title("Q11_2 $E_{val}$",fontsize=20)
# plt.yticks(fontsize=20)
# plt.xticks(fontsize=20)
# plt.xlabel("$log_{10}\lambda$",fontsize=20)
# plt.savefig("Q11_2_bar_plot")
# plt.show()
# plt.close()

plt.figure(figsize=(10.8,10.8))
plt.bar(label11, data11)
plt.title("Q11 $E_{val}$",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel("$log_{10}\lambda$",fontsize=20)
plt.savefig("Q11_bar_plot")
plt.show()
plt.close()

plt.figure(figsize=(10.8,10.8))
plt.bar(label12, data12)
plt.title("Q12 5 Cross Vold",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel("$log_{10}\lambda$",fontsize=20)
plt.savefig("Q12_bar_plot")
plt.show()
plt.close()
