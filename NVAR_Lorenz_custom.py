import Data_Manipulation
import nvar_custom
from Lorenz63 import lorenzfull

from datetime import datetime

def generate_data(length_train, length_test):
    x = lorenzfull(length_train + length_test, 0.01)
    x_train = x[:length_train]
    y_train = x[1:length_train + 1]
    x_test = x[length_train:-1]
    y_test = x[length_train + 1:]
    return x_train, y_train, x_test, y_test


def create_esn(delay = 1, order = 1, strides = 1, ridge_reg=1e-6, ridge_name="default_ridge"):
    nvar = nvar_custom.NVAR(delay=int(delay), order=int(order), strides=int(strides), ridge=ridge_reg)
    return nvar


def experiment(data, delay = 1, order = 1, strides = 1, ridge_reg=1e-6, warmup=100,Plotting = False):
    x_train, y_train, x_test, y_test = data
    my_esn = create_esn(delay=delay, order=order, strides=strides, ridge_reg=ridge_reg, ridge_name=str(delay) + str(order) + str(strides) + str(datetime.now().strftime("%H_%M_%S")))
    my_esn.fit(x_train, y_train, warmup=warmup)
    predictions = my_esn.run(x_test)
    error = Data_Manipulation.error_func_mse(y_test, predictions)

    if Plotting:
        Data_Manipulation.compare_3dData_2dPlot(y_test, predictions)
        Data_Manipulation.compare_3dData_3dPlot(y_test, predictions)

    return error

