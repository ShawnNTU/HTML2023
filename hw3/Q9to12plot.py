import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

rawdata = np.loadtxt("exp_result.txt",skiprows=1)

Q9data,Q10data,Q11Adata,Q11Bdata,Q12Adata,Q12Bdata = np.split(rawdata,indices_or_sections=6,axis=1)

# Q9
start = int(Q9data.min() * 100 / 5) * 5 / 100
stop = (int(Q9data.max() * 100 / 5) + 1) * 5 / 100
plt.figure(figsize=(19.2,10.8))
plt.hist(Q9data,np.arange(start,stop,step=0.01))
plt.yticks(list(range(0,40,5)))
plt.xticks(np.arange(start,stop,step=0.01))
plt.xlabel("$E_{in}^{sqr}(\mathbf{w}_{LIN})$")
plt.title(f"Q9 median:{np.median(Q9data):.4f}")
plt.grid()
plt.savefig("Q9result")
plt.close()

# Q10
start = int(Q10data.min() * 100 / 5) * 5 / 100
stop = (int(Q10data.max() * 100 / 5) + 1) * 5 / 100
plt.figure(figsize=(19.2,10.8))
plt.hist(Q10data,np.arange(start,stop,step=0.005))
plt.yticks(list(range(0,50,5)))
plt.xticks(list(np.arange(start,stop,step=0.005)))
plt.xlabel("$E_{in}^{0/1}(\mathbf{w}_{LIN})$")
plt.title(f"Q10 median:{np.median(Q10data):.4f}")
plt.grid()
plt.savefig("Q10result")
plt.close()

# Q11
plt.figure(figsize=(19.2,10.8))
plt.scatter(Q11Adata,Q11Bdata)
# plt.yticks(np.arange(0,0.6,0.05))
plt.title(f"Q11 A median:{np.median(Q11Adata):.4f},B median:{np.median(Q11Bdata):.4f}")
plt.grid()
plt.xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}))$")
plt.ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}))$")
plt.savefig("Q11result")
plt.close()

# Q12
plt.figure(figsize=(19.2,10.8))
plt.scatter(Q12Adata,Q12Bdata)
# plt.yticks(np.arange(0,0.6,0.05))
plt.title(f"Q12 A median:{np.median(Q12Adata):.4f},B median:{np.median(Q12Bdata):.4f}")
plt.grid()
plt.xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}^{\prime}))$")
plt.ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}^{\prime}))$")
plt.savefig("Q12result")
plt.close()

# added
fig, ax = plt.subplots(2,1,figsize=(12,24))
ax[0].scatter(Q12Adata,Q12Bdata)
ax[0].set_ylim(0.01,0.08)
ax[0].set_xlim(0.01,0.08)
ax[0].set_title(f"Q12")
ax[0].set_xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}^{\prime}))$")
ax[0].set_ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}^{\prime}))$")
ax[0].grid()
ax[1].scatter(Q11Adata,Q11Bdata)
ax[1].set_ylim(0.01,0.08)
ax[1].set_xlim(0.01,0.08)
ax[1].set_title(f"Q11")
ax[1].set_xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}))$")
ax[1].set_ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}))$")
ax[1].grid()
plt.savefig("Added_result")
plt.close()

