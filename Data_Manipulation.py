import numpy as np
from joblib import Parallel, delayed
import sympy
def Array_Combination_to_tuple(array = np.array([],dtype=object),combination = np.array([])):
    tuple = ()
    for i in range(array.shape[0]):
        tuple = tuple + (array[i][combination[i]],)
    return tuple

def Array_to_tuple(array = np.array([],dtype=object)):
    tuple = ()
    for i in range(array.shape[0]):
        tuple = tuple + (array[i],)
    return tuple

def error_func_mse(x, y):
    return np.average(np.power(sum(np.square(x - y)), 0.5))
def array_min_finder(input_array = np.array([0],dtype=object),maxthreads = 1): #output is a tuple with first element is the location vector (as an np.array), second is the minimum value
    if (input_array.ndim == 1):
        return np.array([np.array([np.argmin(input_array)]),np.amin(input_array)],dtype=object)
    else:
        this_dimensions_size = input_array.shape[0]
        sub_solutions = Parallel(n_jobs=min(input_array.size,maxthreads))(delayed(array_min_finder)(input_array[i],maxthreads = max(1,(maxthreads//input_array.size))) for i in range(this_dimensions_size))

        locations = np.full(this_dimensions_size,0,dtype=object)
        minimums = np.full(this_dimensions_size,0,dtype=float)
        for i in range(this_dimensions_size):
            locations[i] = sub_solutions[i][0]
            minimums[i] = sub_solutions[i][1]

        return (np.append(np.array([np.argmin(minimums)]),locations[np.argmin(minimums)]),minimums[np.argmin(minimums)])

def multidimensional_array_special_averaging(array = np.array([],dtype=object), array_IS_this_dimension_needs_averaging = np.array([],dtype=bool)):
    #the averaged dimensions of the array will still be there, but it will be 1 long
    dimensions_to_Average = ()
    for i in range(array_IS_this_dimension_needs_averaging.shape[0]):
        if(array_IS_this_dimension_needs_averaging[i] == True):
            dimensions_to_Average = dimensions_to_Average + (i,)

    for i in dimensions_to_Average:
        array = array.mean(axis = i,keepdims=True)

    return array

def data_out_of_symbols(delay = 0,dimension = 1):   #make a data, but symbols instead of floats

    low_dimension_names = ["X","Y","Z"] #for dimension <4 we use these instead of X1,X2,X3...
    delay_prefix = "P"  #this is added before the name for every delaystep backwards
    delay_suffix = ""   #this is added after  the name for every delaystep backwards

    if dimension > len(low_dimension_names):
        dimension_names = np.full(dimension,"string")
        for i in range(dimension):
            dimension_names[i] = "X" + str(i + 1)
            print(dimension_names[i])
    else:
        dimension_names = low_dimension_names

    symbol_data = np.zeros((delay + 1, dimension), dtype=object)
    for curr_delay in range(delay + 1):
        for curr_dimension in range(dimension):
            symbol_data[curr_delay, curr_dimension] = sympy.symbols(delay_prefix * (delay-curr_delay) + dimension_names[curr_dimension] + delay_suffix * (delay-curr_delay))
    return symbol_data
