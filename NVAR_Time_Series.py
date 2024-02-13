import numpy as np
import Data_Manipulation
import NVAR
import Plots

def make_delayed(data, delay):

    output = np.zeros((data.shape[0],data.shape[1]),dtype=data.dtype)
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
        x_train = delay_data(data = TS_data,delay = self.delay)[self.delay+warmup:-1,:]
        y_train = TS_data[self.delay+warmup+1:,:] - TS_data[self.delay+warmup:-1,:]

        self.NVAR.fit(x_train=x_train,y_train=y_train)
        return

    def predict(self,initial_data = np.array([]), predict_time = 1):

        if initial_data.shape[0] < self.delay:
            return

        dim = initial_data.shape[1]
        initialisation = initial_data.shape[0]

        #Make extended data with the initial data, and add the already known delays:
        y_test = delay_data(np.append(initial_data,np.zeros((predict_time,dim)),axis=0),self.delay)

        for i in range(predict_time):
            diff = self.NVAR.run_on_vector(y_test[initialisation + i - 1,:])
            current_result = y_test[initialisation + i - 1,:dim]  + diff

            #Put current result in correct positions:
            for shift in range(min(self.delay,predict_time-i-1)+1):
                y_test[ (initialisation + i) + shift,dim * shift: dim * (shift+1)] = current_result



        return y_test[initialisation:,:dim]

