import Kombi
import Data_Manipulation
import numpy as np
from joblib import Parallel, delayed
import Plots
import NVAR_Time_Series
import Differential_Equation
import sympy

def best_parameters_finder(
        function,
        parameters_intervals = np.array([],dtype=object),
        threads: int = 4,
        searching_array_size: int = 7,
        parameters_IS_Needs_averageing = np.array([False],dtype=bool), # a "parameters" long bool array, or a 1 long with a False
        goal_error_diff_percentage = 0.01):
    def run_Parallel_on_array(function, array=np.array([], dtype=object), threads=1):
        array_sizes = np.full(array.shape[0], 0, dtype=int)
        for i in range(array.shape[0]):
            array_sizes[i] = array[i].shape[0]
        number_of_runs = 1
        for i in range(array_sizes.size):
            number_of_runs *= array_sizes[i]


        results = Parallel(n_jobs=threads)(delayed(function)(
            *Data_Manipulation.Array_Combination_to_tuple(array, Kombi.kulonbozoSzamjegyuSzam(array_sizes, i))) for i in
                                           range(number_of_runs))
        results = np.array(results).reshape(*Data_Manipulation.Array_to_tuple(array_sizes))
        return results

    def Generate_new_Intervals(current_intervals=np.array([], dtype=object),
                               current_Parameterarray=np.array([], dtype=object),
                               minimum_Lokation=np.array([], dtype=float),
                               array_IS_needs_averaging=np.array([False], dtype=bool)):
        Is_Minimum_coordinate_On_Edge = np.full(minimum_Lokation.shape[0], False, dtype=bool)

        for i in range(minimum_Lokation.shape[0]):
            if (current_Parameterarray[i].shape[0] > 2):
                if (current_intervals[i].shape[0] != 1):
                    if (minimum_Lokation[i] == 0):
                        Is_Minimum_coordinate_On_Edge[i] = True
                    if (minimum_Lokation[i] == current_Parameterarray[i].shape[0]):
                        Is_Minimum_coordinate_On_Edge[i] = True

        if (array_IS_needs_averaging.shape[0] != current_intervals.shape[0]):
            array_IS_needs_averaging = np.full(current_intervals.shape[0], False)

        new_intervals = np.full(current_intervals.shape[0], 0, dtype=object)
        for i in range(minimum_Lokation.shape[0]):
            if ((current_Parameterarray[i].shape[0] > 2) and (array_IS_needs_averaging[i] == False)):
                if Is_Minimum_coordinate_On_Edge[i]:
                    if (minimum_Lokation[i] == 0):
                        new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i]],
                                                     current_Parameterarray[i][minimum_Lokation[i] + 2]])
                    else:
                        new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 2],
                                                     current_Parameterarray[i][minimum_Lokation[i]]])
                else:
                    new_intervals[i] = np.array([current_Parameterarray[i][minimum_Lokation[i] - 1],
                                                 current_Parameterarray[i][minimum_Lokation[i] + 1]])
            else:
                new_intervals[i] = current_intervals[i]
        return new_intervals

    current_intervals = parameters_intervals

    counter = 0
    prev_result = float('inf')
    result_diff = float('inf')


    while((counter < 100) and (result_diff >= (prev_result * goal_error_diff_percentage)) and (result_diff > 0)):
        print("Current error differetial percentage = " + str(result_diff * prev_result))
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
            errors = Data_Manipulation.multidimensional_array_special_averaging(errors, parameters_IS_Needs_averageing)

        #print("Errors: " + str(errors))
        (minimum_Location,minimum_Value) = Data_Manipulation.array_min_finder(errors, maxthreads=threads)
        print("Minimum found: " + str(minimum_Value) + "   In location: " + str(minimum_Location))

        current_intervals = Generate_new_Intervals(current_intervals,current_Parameterarray,minimum_Location,parameters_IS_Needs_averageing)

        result_diff = prev_result - minimum_Value
        prev_result = minimum_Value
        counter += 1
    print("\n\nBest parameters found: \n")
    for i in range(current_Parameterarray.shape[0]):
        if(parameters_IS_Needs_averageing.shape[0] != 1) and (parameters_IS_Needs_averageing[i] == False):
            if(current_Parameterarray[i].shape[0] > 2):
                print(current_Parameterarray[i][minimum_Location[i]])
            else:
                print("This parameter is not changing")
        else:
            print("This parameter is averaged")
    return


saved_runs = {"Paper reproduction": {"Train length" : 600,
                                      "Test length" : 799,
                                      "Equation" : "Lorenz",
                                      "Time step length" : 0.025,
                                      "Method" : "RK23",
                                      "Starting point" : [17.67715816276679, 12.931379185960404, 43.91404334248268],
                                      "Delay" : 1,
                                      "Order" : 2,
                                      "Warmup length" : 198,
                                      "Ridge" : 2.5e-6,
                                      "Plotting" : True,
                                      "Printing" : True
                                      },
               "Paper reproduction 2": {"Train time" : 10.,
                                        "Test time" : 20.,
                                        "Time step length": 0.025,
                                        "Equation" : "Lorenz",
                                        "Method" : "RK23",
                                        "Starting point" : [17.67715816276679, 12.931379185960404, 43.91404334248268],
                                        "Delay" : 1,
                                        "Order" : 2,
                                        "Warmup time" : 5.,
                                        "Ridge" : 2.5e-6,
                                        "Plotting" : True,
                                        "Printing" : True
                                        },
               "Rossler example": {"Train length" : 500,
                                      "Test length" : 2000,
                                      "Equation" : "Rossler",
                                      "Time step length" : 0.025,
                                      "Method" : "RK23",
                                      "Starting point" : [17.67715816276679, 12.931379185960404, 43.91404334248268],
                                      "Delay" : 1,
                                      "Order" : 2,
                                      "Warmup length" : 198,
                                      "Ridge" : 2.5e-6,
                                      "Plotting" : True,
                                      "Printing" : True
                                      },
               "Chua example": {"Train length" : 400,
                                "Test length" : 400,
                                "Equation" : "Chua",
                                "Time step length" : 0.025,
                                "Method" : "RK23",
                                "Starting point" : [0.2,0.1,0.1],
                                "Delay" : 4,
                                "Order" : 3,
                                "Warmup length" : 198,
                                "Ridge" : 0.0000109,
                                "Plotting" : True,
                                "Printing" : True
                                }
               }


def dict_has_NVAR_params(dict):
    if "Delay" not in dict:
        return False
    if "Order" not in dict:
        return False
    if "Ridge" not in dict:
        return False
    return True

def dict_has_data(dict):
    if "Data" in dict:
        if "Train length" in dict:
            return True
    return False

def dict_can_create_data(dict):

    if "Time step length" not in dict:
        dict["Time step length"] = 0.025


    if "Equation" in dict:
        if "Starting point" not in dict:
            return False

        if dict["Equation"] == "Lorenz":
            Current_Equation = Differential_Equation.Lorenz63()
        elif dict["Equation"] == "Rossler":
            Current_Equation = Differential_Equation.Rossler()
        elif dict["Equation"] == "Chua":
            Current_Equation = Differential_Equation.Chua()
        else:
            return False

        if "Method" not in dict:
            dict["Method"] = ""

        if "Train time" in dict:
            if "Test time" in dict:

                dict["Data"] = Current_Equation.generate_data(n_timepoints=int((dict["Train time"] + dict["Test time"])/dict["Time step length"]), dt=dict["Time step length"], x0=dict["Starting point"])
                dict["Train length"] = int(dict["Train time"]/dict["Time step length"])

        elif "Train length" in dict:
            if "Test length" in dict:

                dict["Data"] = Current_Equation.generate_data(n_timepoints=dict["Train length"] + dict["Test length"], dt=dict["Time step length"], x0=dict["Starting point"])

        else:
            return False

        #Initialise Warmup
        if "Warmup time" not in dict:
            if "Warmup length" not in dict:
                dict["Warmup length"] = 0
                dict["Warmup time"] = 0
            else:
                dict["Warmup time"] = dict["Warmup length"] * dict["Time step length"]
        else:
            dict["Warmup length"] = int(dict["Warmup time"] / dict["Time step length"])
    else: return False
    return True


def TS_run_on_dict(dict = {}):         #Still not complete

    if not dict_has_NVAR_params(dict):
        return

    if not dict_has_data(dict):
        if not dict_can_create_data(dict):
            return
    #else the "dict_can_create_data(dict)" will create the data

    x_train = dict["Data"][:dict["Train length"]]
    x_test = dict["Data"][dict["Train length"]:]

    my_nvar = NVAR_Time_Series.Nvar_TS(delay=dict["Delay"],order=dict["Order"],ridge=dict["Ridge"])
    my_nvar.fit(x_train,warmup=dict["Warmup length"])


    initialization = x_train[-dict["Delay"] - 1:]
    predictions = my_nvar.predict(initialization, predict_time=x_test.shape[0])
    error = Data_Manipulation.error_func_mse(x_test, predictions)

    if "Printing" in dict:
        if dict["Printing"]:
            my_nvar.NVAR.debug_print()
            print("Ground truth: ")
            print(x_test)
            print("Predicted data: ")
            print(predictions)

    if "Plotting" in dict:
        if dict["Plotting"]:
            Plots.compare_3dData_2dPlot(x_test, predictions)
            Plots.compare_3dData_3dPlot(x_test, predictions)

            symbol_data = np.zeros((dict["Delay"]+1,len(dict["Starting point"])),dtype=object)
            for i in range(dict["Delay"]+1):
                for j in range(len(dict["Starting point"])):
                    if i == 0:
                        symbol_data[i,j] = sympy.symbols("X" + str(j+1))
                    else:
                        symbol_data[i,j] = sympy.symbols("P"*i + "X" + str(j+1))

            print(symbol_data)

            res = my_nvar.predict(symbol_data)
            print(res)
            labels = ["const"]
            for i in range(27):
                labels += "i"
            Plots.histogram_W_out(my_nvar.NVAR.W_out, labels)

    return error

TS_run_on_dict(saved_runs["Chua example"])