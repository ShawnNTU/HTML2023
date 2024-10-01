import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

rawdata = np.loadtxt("outlier_result.txt",skiprows=1)

Q11Adata,Q11Bdata,Q12Adata,Q12Bdata = np.split(rawdata,indices_or_sections=4,axis=1)

# Q11
plt.figure(figsize=(19.2,10.8))
plt.scatter(Q11Adata,Q11Bdata)
plt.title(f"Q11 A median:{np.median(Q11Adata):.4f},B median:{np.median(Q11Bdata):.4f}")
plt.grid()
plt.xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}))$")
plt.ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}))$")
plt.savefig("Q11_out_result")
plt.close()

# Q12
plt.figure(figsize=(19.2,10.8))
plt.scatter(Q12Adata,Q12Bdata)
plt.title(f"Q12 A median:{np.median(Q12Adata):.4f},B median:{np.median(Q12Bdata):.4f}")
plt.grid()
plt.xlabel("$E_{out}^{0/1}(\mathcal{A}(\mathcal{D}^{\prime}))$")
plt.ylabel("$E_{out}^{0/1}(\mathcal{B}(\mathcal{D}^{\prime}))$")
plt.savefig("Q12_out_result")
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
plt.savefig("Added_out_result")
plt.close()