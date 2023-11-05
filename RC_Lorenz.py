import numpy as np

import Kombi
import Data_Manipulation
from Lorenz63 import lorenzfull

from reservoirpy.nodes import Ridge, Reservoir, NVAR
from reservoirpy import set_seed
from joblib import Parallel, delayed


def generate_data(length_train, length_test):
    x = lorenzfull(length_train + length_test, 0.01)
    x_train = x[:length_train]
    y_train = x[1:length_train + 1]
    x_test = x[length_train:-1]
    y_test = x[length_train + 1:]
    return x_train, y_train, x_test, y_test


def create_esn(reservoir_size, reservoir_lr=0.9, reservoir_sr=0.5, ridge_reg=1e-6, ridge_name="default_ridge"):
    res = Reservoir(reservoir_size, lr=reservoir_lr, sr=reservoir_sr)
    readout = Ridge(ridge=ridge_reg, name=ridge_name)
    deep_esn = res >> readout
    return deep_esn


def experiment(data, reservoir_size, reservoir_lr=0.3, reservoir_sr=1.25, ridge_reg=1e-6, seed=0, warmup=100,Plotting = False):
    x_train, y_train, x_test, y_test = data
    set_seed(int(seed))
    my_esn = create_esn(int(reservoir_size), reservoir_lr=reservoir_lr, reservoir_sr=reservoir_sr, ridge_reg=ridge_reg, ridge_name=str(reservoir_size) + str(seed) + str(reservoir_lr) + str(reservoir_sr))
    my_esn.fit(x_train, y_train, warmup=warmup)
    predictions = my_esn.run(x_test)
    error = Data_Manipulation.error_func_mse(y_test, predictions)

    if Plotting:
        Data_Manipulation.compare_3dData_2dPlot(y_test, predictions)
        Data_Manipulation.compare_3dData_3dPlot(y_test, predictions)

    return error

def experiment_Dummy(data, reservoir_size, reservoir_lr=0.3, reservoir_sr=1.25, ridge_reg=1e-6, seed=0, warmup=100,Plotting = False):
    x_train, y_train, x_test, y_test = data
    return 10

def run_Parallel_on_array(function,array = np.array([],dtype=object),threads = 1):

    array_sizes = np.full(array.shape[0],0,dtype=int)
    for i in range(array.shape[0]):
        array_sizes[i] = array[i].shape[0]

    number_of_runs = Kombi.kulonbozoSzamjegyuSzamokSzama(array_sizes)
    print(type(threads))
    results = Parallel(n_jobs=threads)(delayed(function)(*Data_Manipulation.Array_Combination_to_tuple(array,Kombi.kulonbozoSzamjegyuSzam(array_sizes,i))) for i in range(number_of_runs))

    results = np.array(results).reshape(*Data_Manipulation.Array_to_tuple(array_sizes))
    return results


def Generate_new_Intervals(current_intervals = np.array([],dtype=object),current_Parameterarray = np.array([],dtype=object),minimum_Lokation = np.array([],dtype=float),strict_intervals: bool = True):
    Is_Minimum_coordinate_On_Edge = np.full(minimum_Lokation.shape[0], False, dtype=bool)
    Is_Minimum_On_Edge = False
    for i in range(minimum_Lokation.shape[0]):
        if (current_Parameterarray[i].shape[0] > 2):
            if (current_intervals[i].shape[0] != 1):
                if (minimum_Lokation[i] == 0):
                    Is_Minimum_coordinate_On_Edge[i] = True
                    Is_Minimum_On_Edge = True
                if (minimum_Lokation[i] == current_Parameterarray[i].shape[0]):
                    Is_Minimum_coordinate_On_Edge[i] = True
                    Is_Minimum_On_Edge = True


    new_intervals = np.full(current_intervals.shape[0], 0, dtype=object)
    if strict_intervals:
        for i in range(minimum_Lokation.shape[0]):
            if (current_Parameterarray[i].shape[0] > 2):
                if Is_Minimum_coordinate_On_Edge[i]:
                    if (minimum_Lokation[i] == 0):
                        new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i]],current_Parameterarray[i][minimum_Lokation[i] + 2]])
                    else:
                        new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 2],current_Parameterarray[i][minimum_Lokation[i]]])
                else:
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 1],current_Parameterarray[i][minimum_Lokation[i] + 1]])
            else:
                new_intervals[i] = current_intervals[i]
    else:
        if Is_Minimum_On_Edge:
            for i in range(minimum_Lokation.shape[0]):
                if (current_Parameterarray[i].shape[0] > 2):
                    current_interval_size_half = current_intervals[i][1] - current_intervals[i][0] / 2
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i]] - current_interval_size_half,current_Parameterarray[i][minimum_Lokation[i]] + current_interval_size_half])
                else:
                    new_intervals[i] = current_intervals[i]
        else:
            for i in range(minimum_Lokation.shape[0]):
                if (current_Parameterarray[i].shape[0] > 2):
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 1],current_Parameterarray[i][minimum_Lokation[i] + 1]])
                else:
                    new_intervals[i] = current_intervals[i]
    return new_intervals


def best_parameters_finder(
        function,
        parameters_intervals = np.array([],dtype=object),
        threads: int = 4,
        searching_array_size: int = 7,
        strict_intervals: bool = True,
        goal_error_diff_percentage = 0.01):

    current_intervals = parameters_intervals
    minimum_Lokation = np.array([],dtype=float)
    minimum_Value = 0


    counter = 0
    prev_result = float('inf')
    result_diff = float('inf')
    print("Current error differetial percentage = " +  str(result_diff*prev_result))
    while((counter < 100) and (result_diff >= (prev_result * goal_error_diff_percentage))):
        current_Parameterarray = np.full(current_intervals.shape[0], 0, dtype=object)

        for i in range(current_intervals.shape[0]):
            if (current_intervals[i].shape[0] == 1):
                current_Parameterarray[i] = current_intervals[i]
            else:
                if(((type(current_intervals[i][0]) == type(np.int32(1))) or (type(current_intervals[i][0]) == type(np.float64(1)))) and ((type(current_intervals[i][1]) == type(np.int32(1))) or (type(current_intervals[i][1]) == type(np.float64(1))))):
                    current_Parameterarray[i] = np.linspace(start=current_intervals[i][0], stop=current_intervals[i][1],num=searching_array_size)
                else:
                    print("ERROR, cannot build linspace in parameter number " + str(i))
                    print("It's types are: ")
                    print(type(current_intervals[i][0]))
                    print(type(current_intervals[i][1]))
                    print("but needs to be: ")
                    print(type(np.int32(1)))
                    print(" or ")
                    print(type(np.float64(1)))
                    current_Parameterarray[i] = current_intervals[i]
        print("Running on: " + str(current_intervals))
        errors = run_Parallel_on_array(function,current_Parameterarray,threads)
        print("Errors: " + str(errors))
        (minimum_Lokation,minimum_Value) = Data_Manipulation.array_min_finder(errors, maxthreads=threads)
        print("Minimum found: " + str(minimum_Value) + "   In lokation: " + str(minimum_Lokation))

        current_intervals = Generate_new_Intervals(current_intervals,current_Parameterarray,minimum_Lokation,strict_intervals)

        result_diff = prev_result - minimum_Value
        prev_result = minimum_Value
        counter += 1
    print("\n\nBest parameters found: ")
    for i in range(current_Parameterarray.shape[0]):
        if(current_Parameterarray[i].shape[0] > 2):
            print(current_Parameterarray[i][minimum_Lokation[i]])
        else:
            print("This parameter is not changing")

    return


def tesztFuti():
    x_train, y_train, x_test, y_test = generate_data(3000, 1000)
    data = (x_train, y_train, x_test, y_test)

    datas = np.array([data],dtype=object)
    Reservoir_size_interval = np.array([100, 5000])
    Leaking_Rate_interval = np.array([0.1,0.9])
    Spectral_Radius_interval = np.array([0.1,2])
    ridge_reg_interval = np.array([1e-6])
    seeds = np.array([int(10)],dtype=int)
    warmups = np.array([100])

    running_Parameter_intervals = np.array([datas,Reservoir_size_interval,Leaking_Rate_interval,Spectral_Radius_interval],dtype=object)

    best_parameters_finder(experiment,running_Parameter_intervals,threads=20,searching_array_size=5)

tesztFuti()