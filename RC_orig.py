import numpy as np
import pathlib
import reservoirpy

"""teszt1 = np.full((10,2),0)
teszt2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(teszt1)
print("")
print(teszt2)"""


from joblib import Parallel, delayed
def process(i,j):
    return (i * i) + j

results = Parallel(n_jobs=2)(delayed(process)(i,j) for i in range(10) for j in range(3))
print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]








"""
import time

start = time.time()

print(23*2.3)

end = time.time()
print(end - start)
"""