import numpy as np
import NVAR
import Lorenz63
import Data_Manipulation
def make_delayed(data, delay):
    output = np.zeros((data.shape[0],data.shape[1]))
    output[delay:,:] = data[:-delay,:]
    return output

def delay_data( data = np.array([]), delay=1):
    #Initialise with the undelayed parts:
    out_data = data[:, :]

    #Append with delayed parts:
    for i in range(1, delay+1):
        out_data = np.append(out_data, make_delayed(data, i),axis=1)
    return out_data

class Nvar_TS():

    def __init__(self, delay: int, order: int, ridge: float):
        self.NVAR = NVAR.NVAR(order= order, ridge= ridge)
        self.delay = delay      #delay = 0 is when just current datapoint is given
        self.warmup = 0
        return


    def fit(self,TS_data, warmup=0) -> None:
        self.warmup = warmup

        #Initialise x_train and y_train as differentials
        x_train = delay_data(data = TS_data,delay = self.delay)[warmup:-1,:]
        y_train = TS_data[warmup+1:,:] - TS_data[warmup:-1,:]

        self.NVAR.fit(x_train=x_train,y_train=y_train)
        return

    def predict(self,initial_data = np.array([]), predict_time = 1):
        dim = initial_data.shape[1]
        initialisation = initial_data.shape[0]

        #Make extended data with the initial data, and add the already known delays:
        y_test = delay_data(np.append(initial_data,np.zeros((predict_time,dim)),axis=0),self.delay)

        for i in range(predict_time):
            diff = self.NVAR.runpoint(y_test[initialisation + i - 1,:])
            current_result = y_test[initialisation + i - 1,:dim]  + diff

            #Put current result in correct positions:
            for shift in range(min(self.delay,predict_time-i-1)+1):
                y_test[ (initialisation + i) + shift,dim * shift: dim * (shift+1)] = current_result

        return y_test[initialisation:,:dim]


def Lorenz_prediction(length_train = 1, length_test = 1, delay=1, order=1, ridge_reg=1e-6, warmup=100, Plotting=False):

    data = Lorenz63.lorenzfull(length_train + length_test, 0.025, x0=[17.67715816276679, 12.931379185960404, 43.91404334248268])
    x_train = data[:length_train]
    x_test = data[length_train:]

    my_nvar = Nvar_TS(delay=delay,order=order,ridge=ridge_reg)
    print("NVAR_TS created")
    my_nvar.fit(x_train,warmup=warmup)

    predictions = my_nvar.predict(x_train[-delay-5:],predict_time=x_test.shape[0])
    print("NVAR TS predicted")
    error = Data_Manipulation.error_func_mse(x_test, predictions)

    if Plotting:
        Data_Manipulation.compare_3dData_2dPlot(x_test, predictions)
        Data_Manipulation.compare_3dData_3dPlot(x_test, predictions)

    return error