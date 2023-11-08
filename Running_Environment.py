import RC_Lorenz
import Kombi
import Data_Manipulation

import numpy as np
from joblib import Parallel, delayed
from datetime import datetime


def run_Parallel_on_array(function,array = np.array([],dtype=object),threads = 1):
    array_sizes = np.full(array.shape[0],0,dtype=int)
    for i in range(array.shape[0]):
        array_sizes[i] = array[i].shape[0]
    number_of_runs = Kombi.kulonbozoSzamjegyuSzamokSzama(array_sizes)
    print(type(threads))
    results = Parallel(n_jobs=threads)(delayed(function)(*Data_Manipulation.Array_Combination_to_tuple(array,Kombi.kulonbozoSzamjegyuSzam(array_sizes,i))) for i in range(number_of_runs))
    results = np.array(results).reshape(*Data_Manipulation.Array_to_tuple(array_sizes))
    return results


def Generate_new_Intervals(current_intervals = np.array([],dtype=object),current_Parameterarray = np.array([],dtype=object),minimum_Lokation = np.array([],dtype=float),array_IS_needs_averaging = np.array([False],dtype=bool)):
    Is_Minimum_coordinate_On_Edge = np.full(minimum_Lokation.shape[0], False, dtype=bool)

    for i in range(minimum_Lokation.shape[0]):
        if (current_Parameterarray[i].shape[0] > 2):
            if (current_intervals[i].shape[0] != 1):
                if (minimum_Lokation[i] == 0):
                    Is_Minimum_coordinate_On_Edge[i] = True
                if (minimum_Lokation[i] == current_Parameterarray[i].shape[0]):
                    Is_Minimum_coordinate_On_Edge[i] = True

    if(array_IS_needs_averaging.shape[0] != current_intervals.shape[0]):
        array_IS_needs_averaging = np.full(current_intervals.shape[0],False)


    new_intervals = np.full(current_intervals.shape[0], 0, dtype=object)
    for i in range(minimum_Lokation.shape[0]):
        if ((current_Parameterarray[i].shape[0] > 2) and (array_IS_needs_averaging[i] == False)):
            if Is_Minimum_coordinate_On_Edge[i]:
                if (minimum_Lokation[i] == 0):
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i]],current_Parameterarray[i][minimum_Lokation[i] + 2]])
                else:
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 2],current_Parameterarray[i][minimum_Lokation[i]]])
            else:
                new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 1],current_Parameterarray[i][minimum_Lokation[i] + 1]])
        else:
            new_intervals[i] = current_intervals[i]
    return new_intervals


def best_parameters_finder(
        function,
        parameters_intervals = np.array([],dtype=object),
        threads: int = 4,
        searching_array_size: int = 7,
        parameters_IS_Needs_averageing = np.array([False],dtype=bool), # a "parameters" long bool array, or a 1 long with a False
        goal_error_diff_percentage = 0.01):

    current_intervals = parameters_intervals

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
        print("Running on: ")
        print(current_intervals)
        errors = run_Parallel_on_array(function,current_Parameterarray,threads)

        if((parameters_IS_Needs_averageing.shape[0] != 1) or parameters_IS_Needs_averageing[0] == True):
            errors = Data_Manipulation.multidimensional_array_special_averaging(errors,parameters_IS_Needs_averageing)

        #print("Errors: " + str(errors))
        (minimum_Location,minimum_Value) = Data_Manipulation.array_min_finder(errors, maxthreads=threads)
        print("Minimum found: " + str(minimum_Value) + "   In location: " + str(minimum_Location))

        current_intervals = Generate_new_Intervals(current_intervals,current_Parameterarray,minimum_Location,parameters_IS_Needs_averageing)

        result_diff = prev_result - minimum_Value
        prev_result = minimum_Value
        counter += 1
    print("\n\nBest parameters found: \n")
    for i in range(current_Parameterarray.shape[0]):
        if(current_Parameterarray[i].shape[0] > 2):
            print(current_Parameterarray[i][minimum_Location[i]])
        else:
            print("This parameter is not changing")
    return
def tesztFuti():
    x_train, y_train, x_test, y_test = RC_Lorenz.generate_data(3000, 1000)
    data = (x_train, y_train, x_test, y_test)

    datas = np.array([data],dtype=object)
    Reservoir_size_interval = np.array([100, 200])
    Leaking_Rate_interval = np.array([0.05,0.95])
    Spectral_Radius_interval = np.array([0.1,2])
    ridge_reg_interval = np.array([1e-6])
    seeds = np.array([10,20,30,40,50,60,70,80,90,100])
    warmups = np.array([100])

    running_Parameter_intervals = np.array([datas,Reservoir_size_interval,Leaking_Rate_interval,Spectral_Radius_interval,ridge_reg_interval,seeds,warmups],dtype=object)

    this_parameters_need_averaging = np.array([False,False,False,False,False,True,False])
    best_parameters_finder(RC_Lorenz.experiment,running_Parameter_intervals,threads=4,searching_array_size=5,parameters_IS_Needs_averageing=this_parameters_need_averaging)

tesztFuti()