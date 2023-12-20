import numpy as np
import NVAR
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
            diff = self.NVAR.run_vector(y_test[initialisation + i - 1,:])
            current_result = y_test[initialisation + i - 1,:dim]  + diff

            #Put current result in correct positions:
            for shift in range(min(self.delay,predict_time-i-1)+1):
                y_test[ (initialisation + i) + shift,dim * shift: dim * (shift+1)] = current_result



        return y_test[initialisation:,:dim]


def TS_complete_run(data,trainlength = 200, delay=1, order=1, ridge_reg=2.5e-6, warmup=100, Plotting=False, Printing = False):

    x_train = data[:trainlength]
    x_test = data[trainlength:]

    my_nvar = Nvar_TS(delay=delay,order=order,ridge=ridge_reg)
    my_nvar.fit(x_train,warmup=warmup)


    initialization = x_train[-delay - 1:]
    predictions = my_nvar.predict(initialization, predict_time=x_test.shape[0])
    error = Data_Manipulation.error_func_mse(x_test, predictions)

    if Printing:
        my_nvar.NVAR.debug_print()
        print("Ground truth: ")
        print(x_test)
        print("Predicted data: ")
        print(predictions)

    if Plotting:
        Data_Manipulation.compare_3dData_2dPlot(x_test, predictions)
        Data_Manipulation.compare_3dData_3dPlot(x_test, predictions)
        labels = ["const"]
        for i in range(27):
            labels += "i"
        Data_Manipulation.histogram_W_out(my_nvar.NVAR.W_out,labels)

    return error