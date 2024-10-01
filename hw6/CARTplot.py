import matplotlib.pyplot as plt
import numpy as np

err_in_data = np.genfromtxt("gt_err_in.txt")
err_out_data = np.genfromtxt("gt_err_out.txt")



# Q10 histogram
plt.figure(figsize=(10,5))
plt.hist(err_out_data,bins=np.arange(7,20,1))
plt.xticks(np.arange(7,20,1))
plt.yticks(np.arange(100,1100,100))
plt.grid()
plt.savefig("Q10_plot")
plt.close()

# Q11 scatter plot

plt.figure(figsize=(10,5))
plt.scatter(err_in_data,err_out_data)
plt.scatter([np.mean(err_in_data)],[np.mean(err_out_data)],s=20,c="red")
plt.xticks(np.arange(5,17))
plt.yticks(np.arange(7,20))
plt.grid()
plt.savefig("Q11_plot")
plt.close()


avg_err = []

for i in range(1,2001):
    avg_err.append(np.mean(err_out_data[:i]))
    
plt.figure(figsize=(15,5))
plt.plot(avg_err)
plt.yticks(np.arange(9.5,12.25,0.25))
plt.savefig("Q12_plot")
plt.close()