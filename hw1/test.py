import multiprocessing as mp
import numpy as np
def te(a):
    print(a)

if __name__ == "__main__":
    a = np.array([1,2,3])
    pool = mp.Pool(processes=2)
    pool.map(te,[a] * 10)

    # pool.close()
    # pool.join()
    # pool.map(te,range(1))
    
    
    