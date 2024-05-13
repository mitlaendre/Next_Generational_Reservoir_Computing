import Data_Manipulation
import numpy as np
import Plots
import NVAR_Time_Series
import Differential_Equation
import nevergrad as ng
from concurrent import futures
import sympy

#Making some symbols to work with
x,y,z,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10 = sympy.symbols('x y z x1 x2 x3 x4 x5 x6 x7 x8 x9 x10')
Px,Py,Pz,Px1,Px2,Px3,Px4,Px5,Px6,Px7,Px8,Px9,Px10 = sympy.symbols('Px Py Pz Px1 Px2 Px3 Px4 Px5 Px6 Px7 Px8 Px9 Px10')
PPx,PPy,PPz,PPx1,PPx2,PPx3,PPx4,PPx5,PPx6,PPx7,PPx8,PPx9,PPx10 = sympy.symbols('PPx PPy PPz PPx1 PPx2 PPx3 PPx4 PPx5 PPx6 PPx7 PPx8 PPx9 PPx10')
PPPx,PPPy,PPPz,PPPx1,PPPx2,PPPx3,PPPx4,PPPx5,PPPx6,PPPx7,PPPx8,PPPx9,PPPx10 = sympy.symbols('PPPx PPPy PPPz PPPx1 PPPx2 PPPx3 PPPx4 PPPx5 PPPx6 PPPx7 PPPx8 PPPx9 PPPx10')

saved_runs = {
                "Paper reproduction": {
                    "NVAR":{
                        "Delay" : 1,
                        "Order" : 2,
                        "Warmup length" : 198,
                        "Ridge" : 2.5e-6,
                        "Input_symbols" : [x,y,z]},
                    "Feedback":{
                        "Plotting": {
                            "Enable_plotting": True,
                            "Cutoff_small_weights": 0.,
                            "Figheight" : 8.,
                            "Figwidth" : 8.,
                            "Black_and_white" : False,
                            "Save_image" : False
                        },
                        "Printing": {
                            "Enable_printing": True
                        }
                    },
                    "Data": {
                        "Equation":{
                            "Starting_point" : [17.67715816276679, 12.931379185960404, 43.91404334248268],
                            "Method" : "RK23",
                            "Time_step_length" : 0.0125,
                            "Equation_type" : "Lorenz",
                            "Train_length" : 600,
                            "Test_length" : 799,
                            "Generate_symbolic_W_out" : True
                        }
                    }
                    },
                "Rossler final": {
                    "NVAR": {
                        "Delay": 0,
                        "Order": 2,
                        "Warmup length": 200,
                        "Ridge": 2e-6,
                        "Input_symbols": [x, y, z]
                    },
                    "Feedback": {
                        "Plotting": {
                            "Enable_plotting": True,
                            "Cutoff_small_weights": 0.01,
                            "Figheight": 8.,
                            "Figwidth": 8.,
                            "Black_and_white": False,
                            "Save_image": False
                        },
                        "Printing": {
                            "Enable_printing": True
                        }
                    },
                    "Data": {
                        "Equation": {
                            "Starting_point": [5, 5, 5],
                            "Method": "RK23",
                            "Time_step_length": 0.0125,
                            "Equation_type": "Rossler",
                            "Train_length": 2000,
                            "Test_length": 2000,
                            "Generate_symbolic_W_out": True
                        }
                    }
                },
                "Rossler example": {
                    "NVAR": {
                        "Delay": 1,
                        "Order": 2,
                        "Warmup length": 198,
                        "Ridge": 2.5e-6,
                        "Input_symbols": [x, y, z]
                    },
                    "Feedback": {
                        "Plotting": {
                            "Enable_plotting": True,
                            "Cutoff_small_weights": 0.,
                            "Figheight": 8.,
                            "Figwidth": 8.,
                            "Black_and_white": False,
                            "Save_image": False
                        },
                        "Printing": {
                            "Enable_printing": True
                        }
                    },
                    "Data": {
                        "Equation": {
                            "Starting_point": [17.67715816276679, 12.931379185960404, 43.91404334248268],
                            "Method": "RK23",
                            "Time_step_length": 0.025,
                            "Equation_type": "Rossler",
                            "Train_length": 500,
                            "Test_length": 2000,
                            "Generate_symbolic_W_out": True
                        }
                    }
                },
                "Chua example": {
                    "NVAR": {
                        "Delay": 1,
                        "Order": 3,
                        "Warmup length": 10,
                        "Ridge": 0.01,
                        "Input_symbols": [x, y, z],
                        "Combine_symbols": [x, y, z, x**2, y**2]
                    },
                    "Feedback": {
                        "Plotting": {
                            "Enable_plotting": True,
                            "Cutoff_small_weights": 0.,
                            "Figheight": 8.,
                            "Figwidth": 8.,
                            "Black_and_white": False,
                            "Save_image": False
                        },
                        "Printing": {
                            "Enable_printing": True
                        }
                    },
                    "Data": {
                        "Equation": {
                            "Starting_point": [0.2, 0.1, 0.1],
                            "Method": "Euler",
                            "Time_step_length": 0.025,
                            "Equation_type": "Chua",
                            "Train_length": 1000,
                            "Test_length": 1000,
                            "Generate_symbolic_W_out": True
                        }
                    }
                },
                "Test_single": {
                                "NVAR":{
                                    "Delay" : 1,
                                    "Order" : 2,
                                    "Warmup_length" : 10,
                                    "Ridge" : 0.001,
                                    "Lasso" : 0.,
                                    #"Input_symbols" : [x,y,z],
                                    #"Combine_symbols" : [],
                                    "Norm_data": False,
                                    "Cutoff_small_influences": 0.
                                    },
                                "Feedback":{
                                    "Plotting": {
                                        "Enable_plotting": True,
                                        "Cutoff_small_weights": 0.001,
                                        "Figheight" : 8.,
                                        "Figwidth" : 8.,
                                        "Black_and_white" : False,
                                        "Save_image" : False
                                    },
                                    "Printing": {
                                        "Enable_printing": True
                                    }
                                },
                                "Optimizer":{
                                    "Budget": 100,
                                    "Num_workers": 10,
                                    "Batch_mode": True,
                                    "Verbosity": 2
                                },
                                "Data": {
                                    "Equation":{
                                        "Starting_point" : [0.2,0.1,0.1],
                                        "Method" : "Adams-Bashforth 1",
                                        "Time_step_length" : 0.025,
                                        "Equation_type" : "Chua",
                                        "Train_length" : 1000,
                                        "Test_length" : 500,
                                        "Generate_symbolic_W_out" : True
                                    },
                                    #"TS_data_train": np.full((200,3),1.),
                                    #"TS_data_test": np.full((200,3),8.)
                                }
                        },
    "Ex2DLinear": {
        "NVAR": {
            "Delay": 0,
            "Order": 2,
            "Warmup length": 200,
            "Ridge": 2e-6,
            "Input_symbols": [x, y, z]
        },
        "Feedback": {
            "Plotting": {
                "Enable_plotting": True,
                "Cutoff_small_weights": 0.01,
                "Figheight": 8.,
                "Figwidth": 8.,
                "Black_and_white": False,
                "Save_image": False
            },
            "Printing": {
                "Enable_printing": True
            }
        },
        "Data": {
            "Equation": {
                "Starting_point": [5, 5],
                "Method": "RK23",
                "Time_step_length": 0.0125,
                "Equation_type": "Ex2DLinear",
                "Train_length": 2000,
                "Test_length": 2000,
                "Generate_symbolic_W_out": True
            }
        }
    },
    "Ex3DLinear": {
        "NVAR": {
            "Delay": 0,
            "Order": 2,
            "Warmup length": 200,
            "Ridge": 2e-6,
            "Input_symbols": [x, y, z]
        },
        "Feedback": {
            "Plotting": {
                "Enable_plotting": True,
                "Cutoff_small_weights": 0.01,
                "Figheight": 8.,
                "Figwidth": 8.,
                "Black_and_white": False,
                "Save_image": False
            },
            "Printing": {
                "Enable_printing": True
            }
        },
        "Data": {
            "Equation": {
                "Starting_point": [5, 5, 5],
                "Method": "RK23",
                "Time_step_length": 0.0125,
                "Equation_type": "Ex3DLinear",
                "Train_length": 2000,
                "Test_length": 2000,
                "Generate_symbolic_W_out": True
            }
        }
    },
              }

def generate_equation_data(Equation_type: str, Train_length: int, Test_length: int,  Starting_point: np.array([]), Method = "Euler", Time_step_length = 0.025,**kwargs):

    if Equation_type == "Lorenz":
        Current_Equation = Differential_Equation.Lorenz63()
    elif Equation_type == "Rossler":
        Current_Equation = Differential_Equation.Rossler()
    elif Equation_type == "Chua":
        Current_Equation = Differential_Equation.Chua()
    elif Equation_type == "Ex2DLinear":
        Current_Equation = Differential_Equation.Ex2DLinear()
    elif Equation_type == "Ex3DLinear":
        Current_Equation = Differential_Equation.Ex3DLinear()
    else:
        print("Invalid Equation_type")
        return False

    data = Current_Equation.generate_data(x0=Starting_point, n_timepoints=Train_length + Test_length, dt=Time_step_length, method=Method,**kwargs)
    print("Data generated: ")
    print(data)
    dict_return = {"TS_data_train": data[:Train_length],"TS_data_test": data[Train_length:],"Differential_Equation" : Current_Equation}
    return dict_return

def init_optim_params(dict):
    if "Optimizer" not in dict:
        dict["Optimizer"] = {}
    if "Budget" not in dict["Optimizer"]:
        dict["Optimizer"]["Budget"] = 100
    if "Num_workers" not in dict["Optimizer"]:
        dict["Optimizer"]["Num_workers"] = 10
    if "Verbosity" not in dict["Optimizer"]:
        dict["Optimizer"]["Verbosity"] = 2
    if "Batch_mode" not in dict["Optimizer"]:
        dict["Optimizer"]["Batch_mode"] = True
    return dict

def TS_run_on_dict(dict):
    #Making sure there is some data
    if ("TS_data_train" not in dict["Data"]) or ("TS_data_test" not in dict["Data"]):
        if "Equation" not in dict["Data"]:
            print("No data or equation found")
            return
        else:
            if "Input_symbols" in dict["NVAR"]:
                dict["Data"]["Equation"]["equation_symbols"] = dict["NVAR"]["Input_symbols"]
            dict["Data"] = generate_equation_data(**dict["Data"]["Equation"])

    #Dealing with the optimizer parameters
    all_parameters = {**dict["NVAR"],**dict["Data"],**dict["Feedback"]}
    optim_parameters = {}
    fix_parameters = {}

    for i in all_parameters.keys():
        if issubclass(type(all_parameters[i]),ng.p.Data):   #Decide if it's the optimizer's parameter
            optim_parameters[i] = all_parameters[i]
        else:
            fix_parameters[i] = all_parameters[i]
    parametrization = ng.p.Instrumentation(**optim_parameters)

    dict = init_optim_params(dict)
    try:    #If there is any parameter to optimize
        optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=dict["Optimizer"]["Budget"],num_workers=dict["Optimizer"]["Num_workers"])
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(lambda **kwargs: TS_run(**{**kwargs,**fix_parameters,"Printing":{},"Plotting":{}}), executor=executor, batch_mode=dict["Optimizer"]["Batch_mode"], verbosity=dict["Optimizer"]["Verbosity"]).value[1]
        print("Optimization Finished")

    except ValueError as e: #If there is no parameter to optimize
        print(e)
        print("Executing single run:")
        recommendation = optim_parameters

    #Doing a run an the end (with the only parameters in single run, or with the optimal parameters in optimized run)
    return TS_run(**fix_parameters,**recommendation)


def TS_run(Delay: int, TS_data_train,TS_data_test,Printing = {},Plotting={},**kwargs):

    #Making the TS_NVAR and fitting it
    my_nvar = NVAR_Time_Series.Nvar_TS(**kwargs,Printing = Printing,Plotting = Plotting,Delay=Delay)
    my_nvar.fit(**kwargs,Printing = Printing,Plotting = Plotting,Delay=Delay,TS_data=TS_data_train)

    #Predicting (starting from the end of the train data)
    initialization = TS_data_train[-Delay - 1:]
    predictions = my_nvar.predict(initialization, Predict_time=TS_data_test.shape[0])
    error = Data_Manipulation.error_func_mse(TS_data_test, predictions)

    #Printing and Plotting
    if ("Differential_Equation" in kwargs) and (len(kwargs["Differential_Equation"].symbolic_equation)>0):
        Gen_W_out = Differential_Equation.W_out_generator(my_nvar.NVAR.input_symbols,my_nvar.NVAR.combine_symbols,kwargs["Differential_Equation"].symbolic_equation)
    if ("Enable_printing" in Printing) and Printing["Enable_printing"]:

        my_nvar.NVAR.debug_print()
        if ("Differential_Equation" in kwargs) and (len(kwargs["Differential_Equation"].symbolic_equation)>0):
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
        Plots.multiple_histogram_W_out(multiple_W_out = np.array([my_nvar.NVAR.W_out]),in_labels = my_nvar.input_symbols,out_labels = my_nvar.NVAR.combine_symbols,**Plotting)
        if ("Differential_Equation" in kwargs) and (len(kwargs["Differential_Equation"].symbolic_equation)>0):
            Plots.multiple_histogram_W_out(multiple_W_out= np.array([Gen_W_out,my_nvar.NVAR.W_out]),in_labels= my_nvar.input_symbols,out_labels = my_nvar.NVAR.combine_symbols,**Plotting)
    return error


TS_run_on_dict(saved_runs["Test_single"])