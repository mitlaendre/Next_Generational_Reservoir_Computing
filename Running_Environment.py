import Data_Manipulation
import numpy as np
from joblib import Parallel, delayed
import Plots
import NVAR_Time_Series
import Differential_Equation
import itertools
import nevergrad as ng
from concurrent import futures


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
               "Chua example": {
                                "NVAR":{
                                    "Delay" : ng.p.Scalar(lower=0, upper=5).set_integer_casting(),
                                    "Order" : 3,
                                    "Warmup length" : 10,
                                    "Ridge" : 0.1
                                    },

                                "Equation": {
                                    "Starting point" : [0.2,0.1,0.1],
                                    "Method" : "Midpoint",
                                    "Time step length" : 0.025,
                                    "Equation type" : "Chua",
                                    "Train length" : 10000,
                                    "Test length" : 1000
                                    },
                                "Other": {
                                    "Plotting": False,
                                    "Printing": True,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.
                                }#,
                                #"Data":{
                                #    "X_train" : np.array([])
                                #    "Y_train" : np.array([])
                                #}


               },
                "Test": {
                                "NVAR":{
                                    "Delay" : 1,
                                    "Order" : 1,
                                    "Warmup length" : 10,
                                    #"Ridge" : ng.p.Scalar(lower=0., upper=1.)
                                    "Ridge" : 0.5
                                    },
                                "Other": {
                                    "Plotting": False,
                                    "Printing": False,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.1
                                    },
                                "Data": {
                                    "X_train": np.full((200,1),1.),
                                    "X_test": np.full((200,1),8.)
                                }
                        }
              }


def dict_has_NVAR_params(dict):
    if "NVAR" not in dict:
        return False
    if "Delay" not in dict["NVAR"]:
        return False
    if "Order" not in dict["NVAR"]:
        return False
    if "Ridge" not in dict["NVAR"]:
        return False
    return True


def dict_has_data(dict):
    if ("Data" in dict) and ("X_train" in dict["Data"]) and ("X_test" in dict["Data"]):
        return True
    return False

def dict_equation_generate_data(dict):

    if "Time step length" not in dict:
        dict["Time step length"] = 0.025


    if "Equation type" in dict:
        if "Starting point" not in dict:
            return False

        if dict["Equation type"] == "Lorenz":
            Current_Equation = Differential_Equation.Lorenz63()
        elif dict["Equation type"] == "Rossler":
            Current_Equation = Differential_Equation.Rossler()
        elif dict["Equation type"] == "Chua":
            Current_Equation = Differential_Equation.Chua()
        else:
            return False

        if "Method" not in dict:
            dict["Method"] = ""

        if "Train time" in dict:
            if "Test time" in dict:

                dict["Data"] = Current_Equation.generate_data(x0=dict["Starting point"], n_timepoints=int(
                    (dict["Train time"] + dict["Test time"]) / dict["Time step length"]), dt=dict["Time step length"],
                                                              method=dict["Method"])
                dict["Train length"] = int(dict["Train time"]/dict["Time step length"])

        elif "Train length" in dict:
            if "Test length" in dict:

                dict["Data"] = Current_Equation.generate_data(x0=dict["Starting point"],
                                                              n_timepoints=dict["Train length"] + dict["Test length"],
                                                              dt=dict["Time step length"], method=dict["Method"])

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

def dict_initialise_other_parameters(dict):
    if "Other" not in dict:
        dict["Other"] : {}
    if "Norm data" not in dict["Other"]:
        dict["Norm data"] = False
    if "Printing" not in dict["Other"]:
        dict["Printing"] = False
    if "Plotting" not in dict["Other"]:
        dict["Plotting"] = False
    if "Cutoff small weights" not in dict["Other"]:
        dict["Cutoff small weights"] = 0.

def TS_run_on_dict(dict = {}):         #Still not complete

    if dict_has_data(dict):
        x_train = dict["Data"]["X_train"]
        x_test = dict["Data"]["X_test"]
    else:
        if "Equation" not in dict:
            return #no data and no equation to begin with
        elif not dict_equation_generate_data(dict["Equation"]): #try to generate data
            return

        x_train = dict["Equation"]["Data"][:dict["Equation"]["Train length"]]
        x_test = dict["Equation"]["Data"][dict["Equation"]["Train length"]:]

    dict_initialise_other_parameters(dict)
    if not dict_has_NVAR_params(dict):
        return

    parametrization = ng.p.Instrumentation(
        delay = dict["NVAR"]["Delay"],
        order = dict["NVAR"]["Order"],
        ridge = dict["NVAR"]["Ridge"],
        #TS_data_train=x_train,
        #TS_data_test=x_test,
        warmup=dict["NVAR"]["Warmup length"],
        norm_data=dict["Other"]["Norm data"],
        Printing=dict["Other"]["Printing"],
        Plotting=dict["Other"]["Plotting"],
        Cutoff_small_weights=dict["Other"]["Cutoff small weights"]
    )
    def TS_run_fixed_params(delay,order,ridge,warmup,norm_data,Cutoff_small_weights): return TS_run(delay=delay,order=order,ridge=ridge,TS_data_test=x_test,TS_data_train=x_train,warmup=warmup,norm_data=norm_data,Printing=False,Plotting=False,Cutoff_small_weights=Cutoff_small_weights)

    try:
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100,num_workers=10)
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(TS_run_fixed_params, executor=executor, batch_mode=True, verbosity=2)

    except ValueError as e:
        print(e)
        print("Executing single run:")
        return TS_run(delay = dict["NVAR"]["Delay"],
        order = dict["NVAR"]["Order"],
        ridge = dict["NVAR"]["Ridge"],
        TS_data_train=x_train,
        TS_data_test=x_test,
        warmup=dict["NVAR"]["Warmup length"],
        norm_data=dict["Other"]["Norm data"],
        Printing=dict["Other"]["Printing"],
        Plotting=dict["Other"]["Plotting"],
        Cutoff_small_weights=dict["Other"]["Cutoff small weights"])
    return  recommendation

def TS_run(delay: int, order: int, ridge: float, TS_data_train,TS_data_test,warmup=0, norm_data = False, Printing = False, Plotting = False, Cutoff_small_weights = 0.):
    my_nvar = NVAR_Time_Series.Nvar_TS(delay=delay, order=order, ridge=ridge)
    my_nvar.fit(TS_data_train, warmup=warmup, norm_data=norm_data, cutoff_small_weights=Cutoff_small_weights)

    initialization = TS_data_train[-delay - 1:]
    predictions = my_nvar.predict(initialization, predict_time=TS_data_test.shape[0])
    error = Data_Manipulation.error_func_mse(TS_data_test, predictions)

    if Printing:
        my_nvar.NVAR.debug_print()
        print("Ground truth: ")
        print(TS_data_test)
        print("Predicted data: ")
        print(predictions)
        print("Symbolic prediction: ")
        print(my_nvar.get_symbolic_prediction())

    if Plotting:
        Plots.compare_3dData_2dPlot(TS_data_test, predictions)
        Plots.compare_3dData_3dPlot(TS_data_test, predictions)

        labels = my_nvar.get_list_of_symbols()
        Plots.histogram_W_out(my_nvar.NVAR.W_out, labels)

    return error


TS_run_on_dict(saved_runs["Test"])