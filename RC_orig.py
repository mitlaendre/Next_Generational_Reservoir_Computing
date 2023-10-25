import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib.pyplot
import reservoirpy


#np.array használata
"""
teszt1 = np.full((10,2),0)
teszt2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(teszt1)
print("")
print(teszt2)"""

"""
#Paralell és egy nagy tömb kezelése

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
"""



#surface plot for errors
"""
tomb = np.array(
                [[[ 58.15923757,  64.4747462,   90.53759092, 113.50645819, 168.68428256],
                  [ 90.97613044,  84.64348166,  82.05740897,  80.47934262,  81.86836333],
                  [112.70567593, 107.54013518, 105.62494151,  98.99207874,  90.34202357],
                  [125.74188876, 122.31270676, 118.16988091, 112.9041992,  108.98378016],
                  [132.83822029, 130.95516304, 126.76384502, 122.85519774, 125.81939769]],

                 [[ 18.70495611,  28.05702499,  63.78762039, 106.34228975, 187.01847576],
                  [ 14.53528095,  23.32955401,  26.84213527,  37.16252606,  43.25204842],
                  [ 28.47863387,  33.24142741,  42.47417412,  72.54227207, 105.75431941],
                  [ 37.12749575,  49.97675714,  74.12889605, 136.41838343, 337.76633145],
                  [ 45.05699719,  47.62938513,  95.05599729, 251.87652286, 409.35448071]],

                 [[ 21.29363039,  27.17554758,  73.97252802, 150.40597075, 276.60478843],
                  [ 25.65613492,  27.16656747,  27.96123024,  36.99924734,  43.84407999],
                  [ 27.27305612,  36.18156575,  55.57702243,  57.5072704,   75.53505627],
                  [ 33.20185763,  40.53210217,  75.3829493, 136.42583232, 143.50443268],
                  [ 38.7618769,   48.31108057,  88.69132182, 239.97998492, 341.20780761]]])

Reservoir_sizes = np.array([100,1000,2000])
Leaking_Rates = np.linspace(start = 0.1, stop = 0.9, num= 5)
Spectral_Radiuses = np.linspace(start = 0.5, stop = 1.5, num= 5)
Seeds = np.array([0,10,20])

def plot_errors_surface(input_errors = np.array([]),Reservoir_sizes = np.array([]),Leaking_Rates = np.array([]),Spectral_Radiuses = np.array([])):
    for i in range(input_errors.shape[0]):
        hf = plt.pyplot.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(Leaking_Rates, Spectral_Radiuses)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_surface(X, Y, input_errors[i])
        ha.set_xlabel('$Leaking Rate$')
        ha.set_ylabel('$Spectral Radius$')
        ha.set_zlabel(r'$Average error$')
        plt.pyplot.show()


plot_errors_surface(tomb,Leaking_Rates = Leaking_Rates,Spectral_Radiuses=Spectral_Radiuses,Reservoir_sizes=Reservoir_sizes)"""


