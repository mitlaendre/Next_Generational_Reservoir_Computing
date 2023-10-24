import numpy as np
import pathlib
import reservoirpy



#np.array használata
"""
teszt1 = np.full((10,2),0)
teszt2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(teszt1)
print("")
print(teszt2)"""


#Paralell és egy nagy tömb kezelése
"""
from joblib import Parallel, delayed
def process(i,j,k,l):
    return i*1000 + j*100 + k*10 + l

results = Parallel(n_jobs=5)(delayed(process)(i,j,k,l) for i in range(2) for j in range(3) for k in range(4) for l in range(5))
print(results)
print("==================================================================")
results1 = np.array(results).reshape(2,3,4,5)
print(results1)
print("==================================================================")
results2 = sum(results1) #i =0,1 adta össze
print(results2)
print("==================================================================")
print(results2[2,3,4]) #i-re osszeadva, j = 2, k = 3, l = 4 elem kiirat
"""


#idő kezelése
""" 
import time

start = time.time()

print(23*2.3)

end = time.time()
print(end - start)
"""


#félkész array min index kereső
"""
x = np.array([[3,2,3],[4,5,6],[7,1,9]])
k=np.argmin(x)

print(k)

a = x
ment = np.full(x.ndim,0)
for i in range(x.ndim):
    ment[i] = k/a.size
    k %= a.size
    a = a[0]

print(k)
print(ment)
"""


#rekurzív parralel array min index kereső
"""
from joblib import Parallel, delayed
def array_min_finder(input_array = np.array([0],dtype=object)): #output is a np.array with first element is the location vector, second is the minimum value
    #print("run with input:")
    #print(input_array)
    if (input_array.ndim == 1):
        return np.array([np.array([np.argmin(input_array)]),np.amin(input_array)],dtype=object)
    else:
        this_dimensions_size = input_array.shape[0]
        sub_solutions = Parallel(n_jobs=input_array.size)(delayed(array_min_finder)(input_array[i]) for i in range(this_dimensions_size))

        locations = np.full(this_dimensions_size,0,dtype=object)
        minimums = np.full(this_dimensions_size,0)
        for i in range(this_dimensions_size):
            locations[i] = sub_solutions[i][0]
            minimums[i] = sub_solutions[i][1]

        return np.array([np.append(np.array([np.argmin(minimums)]),locations[np.argmin(minimums)]),minimums[np.argmin(minimums)]],dtype=object)



x = np.array([[[5,5,5],[5,5,5],[5,1,5]],[[5,5,5],[5,5,5],[5,0,5]]])

print(x)
print("")
sol = array_min_index(x)
print(sol[0])
print(sol[1])
"""
