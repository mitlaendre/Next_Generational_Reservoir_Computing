import Data_Manipulation
import numpy as np
import Plots
import NVAR_Time_Series
import Differential_Equation
import nevergrad as ng
from concurrent import futures
import sympy

x = sympy.symbols('x')
y = sympy.symbols('y')
z = sympy.symbols('z')

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
                                    "Delay" : 1,
                                    "Order" : 3,
                                    "Warmup length" : 10,
                                    "Ridge" : 0.01,
                                    "Input symbols" : [x,y,z],
                                    "Combine symbols" : [x,y,z,x**2,y**2]
                                    },

                                "Equation": {
                                    "Starting point" : [0.2,0.1,0.1],
                                    "Method" : "Euler",
                                    "Time step length" : 0.025,
                                    "Equation type" : "Chua",
                                    "Train length" : 1000,
                                    "Test length" : 1000
                                    },
                                "Other": {
                                    "Plotting": True,
                                    "Printing": True,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.,
                                    "Cutoff W_out": 0.01
                                }#,
                                #"Data":{
                                #    "X_train" : np.array([])
                                #    "Y_train" : np.array([])
                                #}


               },
                "Test_optim": {
                                "NVAR":{
                                    "Delay" : 1,
                                    "Order" : 1,
                                    "Warmup length" : 10,
                                    "Ridge" : ng.p.Scalar(lower=0., upper=1.),
                                    "Input symbols" : [x,y,z],
                                    "Combine symbols" : [1,x,y,z,x**2,y**2]
                                    },
                                "Other": {
                                    "Plotting": True,
                                    "Printing": True,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.1
                                    },
                                "Data": {
                                    "X_train": np.full((200,1),1.),
                                    "X_test": np.full((200,1),8.)
                                },
                                "Optimizer":{
                                    "Budget": 100,
                                    "Num_workers": 10,
                                    "Batch mode": True,
                                    "Verbosity": 2
                                }
                        },
                "Test_single": {
                                "NVAR":{
                                    "Delay" : 1,
                                    "Order" : 1,
                                    "Warmup length" : 10,
                                    "Ridge" : 0.5
                                    },
                                "Other": {
                                    "Plotting": True,
                                    "Printing": True,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.1
                                    },
                                "Data": {
                                    "X_train": np.full((200,3),1.),
                                    "X_test": np.full((200,3),8.)
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
    if "Input symbols" not in dict["NVAR"]:
        dict["NVAR"]["Input symbols"] = []
    if "Combine symbols" not in dict["NVAR"]:
        dict["NVAR"]["Combine symbols"] = []

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
    if "Cutoff W_out" not in dict["Other"]:
        dict["Other"]["Cutoff W_out"] = 0.

    if "Optimizer" not in dict:
        dict["Optimizer"] = {}
    if "Budget" not in dict["Optimizer"]:
        dict["Optimizer"]["Budget"] = 100
    if "Num_workers" not in dict["Optimizer"]:
        dict["Optimizer"]["Num_workers"] = 10
    if "Batch mode" not in dict["Optimizer"]:
        dict["Optimizer"]["Batch mode"] = True
    if "Verbosity" not in dict["Optimizer"]:
        dict["Optimizer"]["Verbosity"] = 0



def TS_run_on_dict(dict = {}):

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

    all_parameters = {
        "delay" : dict["NVAR"]["Delay"],
        "order" : dict["NVAR"]["Order"],
        "ridge" : dict["NVAR"]["Ridge"],
        "TS_data_train" : x_train,
        "TS_data_test" : x_test,
        "input_symbols" : dict["NVAR"]["Input symbols"],
        "combine_symbols" : dict["NVAR"]["Combine symbols"],
        "warmup" : dict["NVAR"]["Warmup length"],
        "norm_data" : dict["Other"]["Norm data"],
        "Cutoff_small_weights" : dict["Other"]["Cutoff small weights"],
        "Cutoff_W_out_plot" : dict["Other"]["Cutoff W_out"]
    }
    fix_parameters = {}
    optim_parameters = {}
    for i in all_parameters.keys():
        if i != "Printing" and i != "Plotting":
            if issubclass(type(all_parameters[i]),ng.p.Data):
                optim_parameters[i] = all_parameters[i]
            else:
                fix_parameters[i] = all_parameters[i]

    parametrization = ng.p.Instrumentation(**optim_parameters)


    try:
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=dict["Optimizer"]["Budget"],num_workers=dict["Optimizer"]["Num_workers"])
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(lambda **kwargs: TS_run(**kwargs,**fix_parameters), executor=executor, batch_mode=dict["Optimizer"]["Batch mode"], verbosity=dict["Optimizer"]["Verbosity"]).value[1]

    except ValueError as e:
        print(e)
        print("Executing single run:")
        recommendation = optim_parameters

    return TS_run(**fix_parameters,**recommendation,Plotting=dict["Other"]["Plotting"],Printing=dict["Other"]["Printing"])

def TS_run(delay: int, order: int, ridge: float, TS_data_train,TS_data_test,warmup=0, norm_data = False, Printing = False, Plotting = False, Cutoff_small_weights = 0., Cutoff_W_out_plot = 0.,input_symbols = None, combine_symbols = None):

    dict = {"NVAR" : {}}
    dict["TS_data"] = TS_data_train

    dict["NVAR"]["Delay"] = delay
    dict["NVAR"]["Order"] = order
    dict["NVAR"]["Ridge"] = ridge

    dict["NVAR"]["Norm data"] = norm_data
    dict["NVAR"]["Cutoff small weights"] = Cutoff_small_weights
    dict["NVAR"]["Input symbols"] = input_symbols
    dict["NVAR"]["Combine symbols"] = combine_symbols

    my_nvar = NVAR_Time_Series.Nvar_TS(dict = dict)
    my_nvar.fit(dict = dict)

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
        Plots.histogram_W_out(my_nvar.NVAR.W_out, labels,cutoff_small_weights=Cutoff_W_out_plot)

    return error


#TS_run_on_dict(saved_runs["Test_optim"])
TS_run_on_dict(saved_runs["Chua example"])