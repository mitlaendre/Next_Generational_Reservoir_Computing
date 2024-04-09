import Data_Manipulation
import numpy as np
import Plots
import NVAR_Time_Series
import Differential_Equation
import nevergrad as ng
from concurrent import futures
import sympy

x,y,z = sympy.symbols('x y z')

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
                                    "Plotting": False,
                                    "Printing": True,
                                    "Norm data": False,
                                    "Cutoff small weights": 0.,
                                    "Cutoff W_out": 0.01
                                },
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
                        },
                "Test_single2": {
                                "NVAR":{
                                    "Delay" : 1,
                                    "Order" : 2,
                                    "Warmup_length" : 10,
                                    "Ridge" : 0.5,
                                    "Input_symbols" : [x,y,z],
                                    "Combine_symbols" : [x,y,z,x**2,y**2,z**2,x**3,y**3,z**3],
                                    "Norm_data": False,
                                    "Cutoff_small_weights": 0.
                                    },
                                "Feedback":{
                                    "Plotting": {
                                        "Enable_plotting": True,
                                        "Cutoff_small_weights": 0.
                                    },
                                    "Printing": {
                                        "Enable_printing": True
                                    }
                                },
                                "Optimizer":{
                                    "Budget": 100,
                                    "Num_workers": 10,
                                    "Batch mode": True,
                                    "Verbosity": 2
                                },
                                "Data": {
                                    "Equation":{
                                        "Starting_point" : [0.2,0.1,0.1],
                                        "Method" : "Euler",
                                        "Time_step_length" : 0.025,
                                        "Equation_type" : "Chua",
                                        "Train_length" : 1000,
                                        "Test_length" : 1000,
                                        "Generate_symbolic_W_out" : True
                                    },
                                    #"TS_data_train": np.full((200,3),1.),
                                    #"TS_data_test": np.full((200,3),8.)
                                }
                        }
              }

def dict_has_data(dict):
    if ("TS_data_train" in dict["Data"]) and ("TS_data_test" in dict["Data"]):
        return True
    return False

def generate_equation_data(Equation_type: str, Train_length: int, Test_length: int,  Starting_point: np.array([]), Method = "Euler", Time_step_length = 0.025,**kwargs):

    if Equation_type == "Lorenz":
        Current_Equation = Differential_Equation.Lorenz63()
    elif Equation_type == "Rossler":
        Current_Equation = Differential_Equation.Rossler()
    elif Equation_type == "Chua":
        Current_Equation = Differential_Equation.Chua()
    else:
        print("Invalid Equation_type")
        return False

    data = Current_Equation.generate_data(x0=Starting_point, n_timepoints=Train_length + Test_length, dt=Time_step_length, method=Method)
    dict_return = {"TS_data_train": data[:Train_length],"TS_data_test": data[Train_length:],"Differential_Equation" : Current_Equation}
    return dict_return


def TS_run_on_dict(dict = {}):

    if not dict_has_data(dict):
        if "Equation" not in dict["Data"]:
            print("No data or equation")
            return
        else:
            dict["Data"] = generate_equation_data(**dict["Data"]["Equation"])

    all_parameters = {**dict["NVAR"],**dict["Data"],**dict["Feedback"]}
    optim_parameters = {}
    fix_parameters = {}

    for i in all_parameters.keys():
        if issubclass(type(all_parameters[i]),ng.p.Data):
            optim_parameters[i] = all_parameters[i]
        else:
            fix_parameters[i] = all_parameters[i]

    parametrization = ng.p.Instrumentation(**optim_parameters)


    try:
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=dict["Optimizer"]["Budget"],num_workers=dict["Optimizer"]["Num_workers"])
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(lambda **kwargs: TS_run(**{**kwargs,**fix_parameters,"Printing":{},"Plotting":{}}), executor=executor, batch_mode=dict["Optimizer"]["Batch mode"], verbosity=dict["Optimizer"]["Verbosity"]).value[1]
        print("Optimization Finished")

    except ValueError as e:
        print(e)
        print("Executing single run:")
        recommendation = optim_parameters

    return TS_run(**fix_parameters,**recommendation)



def TS_run(Delay: int, TS_data_train,TS_data_test,Printing = {},Plotting={},**kwargs):

    my_nvar = NVAR_Time_Series.Nvar_TS(**kwargs,Delay=Delay)
    my_nvar.fit(**kwargs,Delay=Delay,TS_data=TS_data_train)

    initialization = TS_data_train[-Delay - 1:]
    predictions = my_nvar.predict(initialization, predict_time=TS_data_test.shape[0])
    error = Data_Manipulation.error_func_mse(TS_data_test, predictions)

    if ("Differential_Equation" in kwargs):
        Gen_W_out = Differential_Equation.W_out_generator(kwargs["Input_symbols"], kwargs["Combine_symbols"],kwargs["Differential_Equation"].right_side(sympy.symbols("t"),kwargs["Input_symbols"]),kwargs["Differential_Equation"].dt)

    if ("Enable_printing" in Printing) and Printing["Enable_printing"]:

        my_nvar.NVAR.debug_print()
        if "Differential_Equation" in kwargs:
            print("Generator W_out matrix: ")
            print(Gen_W_out)
        print("Ground truth: ")
        print(TS_data_test)
        print("Predicted data: ")
        print(predictions)
        print("Symbolic prediction: ")
        print(my_nvar.NVAR.W_out @ my_nvar.NVAR.combine_symbols)

    if ("Enable_plotting" in Plotting) and Plotting["Enable_plotting"]:
        Plots.compare_3dData_2dPlot(TS_data_test, predictions)
        Plots.compare_3dData_3dPlot(TS_data_test, predictions)
        Plots.histogram_W_out(my_nvar.NVAR.W_out, my_nvar.NVAR.combine_symbols,**Plotting)
        if "Differential_Equation" in kwargs:
            Plots.histogram_W_out(Gen_W_out,my_nvar.NVAR.combine_symbols,**Plotting)
            Plots.compare_histogram_W_out(Gen_W_out,my_nvar.NVAR.W_out,my_nvar.NVAR.combine_symbols,**Plotting)
    return error


#TS_run_on_dict(saved_runs["Test_optim"])
TS_run_on_dict(saved_runs["Test_single2"])