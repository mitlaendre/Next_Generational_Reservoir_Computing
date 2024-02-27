import numpy as np

import Data_Manipulation
import NVAR
import sympy

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


    def fit(self,TS_data, warmup=0, norm_data = False, cutoff_small_weights = 0.) -> None:
        self.warmup = warmup

        self.dim = TS_data.shape[1]

        #Initialise x_train and y_train as differentials
        x_train = delay_data(data = TS_data,delay = self.delay)[self.delay+warmup:-1,:]
        y_train = TS_data[self.delay+warmup+1:,:] - TS_data[self.delay+warmup:-1,:]

        self.NVAR.fit(x_train=x_train,y_train=y_train, norm_data= norm_data,cutoff_small_weights=cutoff_small_weights)
        return

    def predict(self,initial_data = np.array([]), predict_time = 1):

        if initial_data.shape[0] < self.delay:
            return
        if self.dim != initial_data.shape[1]:
            return


        initialisation = initial_data.shape[0]

        #Make extended data with the initial data, and add the already known delays:
        y_test = delay_data(np.append(initial_data,np.zeros((predict_time,self.dim)),axis=0),self.delay)

        for i in range(predict_time):
            diff = self.NVAR.run_on_vector(y_test[initialisation + i - 1,:])
            current_result = y_test[initialisation + i - 1,:self.dim]  + diff

            #Put current result in correct positions:
            for shift in range(min(self.delay,predict_time-i-1)+1):
                y_test[ (initialisation + i) + shift,self.dim * shift: self.dim * (shift+1)] = current_result



        return y_test[initialisation:,:self.dim]

    def get_symbolic_prediction(self):
        data_out_of_symbols = Data_Manipulation.data_out_of_symbols(delay = self.delay,dimension=self.dim)
        return self.predict(data_out_of_symbols)

    def get_list_of_symbols(self):
        #data, but symbols instead of actual floats
        data_out_of_symbols = Data_Manipulation.data_out_of_symbols(delay=self.delay, dimension=self.dim)
        #make the same delay as in prediction
        delayed_data_out_of_symbols = delay_data(data_out_of_symbols, self.delay)[-1]
        #make the combinations as in the NVAR
        combined_data_out_of_symbols = NVAR.combine_data(delayed_data_out_of_symbols, order=self.NVAR.order)
        combined_data_out_of_symbols[0] = sympy.symbols("Constant")
        return combined_data_out_of_symbols
