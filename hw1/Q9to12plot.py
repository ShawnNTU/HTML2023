import numpy as np
import matplotlib.pyplot as plt
result = np.genfromtxt("result.txt")
mini = int(np.min(result)/100) * 100
maxi = (int(np.max(result)/100) + 1) * 100

bin = range(mini,maxi,100)
fig,axes = plt.subplots(2,2,figsize=(19.2,10.8),layout='tight')
axes = np.ravel(axes)

for i,ax in enumerate(axes):
    ax.hist(x=result[i],bins=bin)
    ax.set_xticks(bin)
    ax.tick_params(labelsize=18)
    ax.set_title(f"Q{9+i}, median={np.median(result[i])}",fontsize=18)
    ax.grid(True)
plt.savefig("Q9toQ12Result")
