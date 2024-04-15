import numpy as np
import Data_Manipulation
import NVAR
import sympy

def delay_data( data = np.array([]), delay=1):
    def make_delayed(data, delay):
        output = np.zeros((data.shape[0], data.shape[1]), dtype=data.dtype)
        output[delay:, :] = data[:-delay, :]
        return output

    #Initialise with the undelayed parts:
    out_data = data[:, :]

    #Append with delayed parts:
    for i in range(1, delay+1):
        out_data = np.append(out_data, make_delayed(data, i),axis=1)
    return out_data

class Nvar_TS():

    def __init__(self,Delay: int,**kwargs):

        self.NVAR = NVAR.NVAR(**kwargs)
        self.delay = Delay      #delay = 0 is when just current datapoint is given
        self.warmup = 0
        self.input_symbols = None   #not contains delayes
        self.combine_symbols = None
        return


    def fit(self,TS_data: np.array([]), Warmup = 0,Input_symbols = [],Combine_symbols = [], **kwargs) -> None:

        self.warmup = Warmup
        self.dim = TS_data.shape[1]
        self.input_symbols = Input_symbols
        self.combine_symbols = Combine_symbols

        #Initialise x_train and y_train as differentials
        X_train = delay_data(data = TS_data,delay = self.delay)[self.delay + Warmup:-1,:]
        Y_train = TS_data[self.delay + Warmup+1:,:] - TS_data[self.delay + Warmup:-1,:]

        #Input_symbols init.
        if len(Input_symbols) != 0: #Input_symbols is given
            if len(Input_symbols) != TS_data.shape[1]: #if not matching length
                if ("Printing" in kwargs) and ("Enable_printing" in kwargs["Printing"]) and kwargs["Printing"]["Enable_printing"]:  print("Input_symbols length not matching, using the default")
                Input_symbols = Data_Manipulation.data_out_of_symbols(0,dimension=self.dim)[0]
                self.combine_symbols = []
            undelayed_symbols = Input_symbols
            Input_symbols = Data_Manipulation.data_out_of_symbols(self.delay, input_symbols=Input_symbols).flatten()   #delay it

        #Combine_symbols init.
        #check if any delayed (Px,PPx,...) variables are there
        contains_delayed_vars = False
        for i in range(len(self.combine_symbols)):
            for j in range(len(undelayed_symbols)):
                for k in range(1,100):
                    temp = sympy.symbols('P'*k + undelayed_symbols[j].name)
                    if temp in Combine_symbols[i].free_symbols: #if it contains it in any way
                        contains_delayed_vars = True
                        break
                if contains_delayed_vars: break
            if contains_delayed_vars:break
        #If there are no delayed parts in it, then add the delayed parts
        all_usable_symbols =  Data_Manipulation.data_out_of_symbols(self.delay, input_symbols=undelayed_symbols)
        if not contains_delayed_vars:
            for curr_delay in range(all_usable_symbols.shape[0]-1):
                for num_symbol in range(len(self.combine_symbols)):
                    self.combine_symbols = self.combine_symbols + [self.combine_symbols[num_symbol].subs(dict(zip(all_usable_symbols[-1],all_usable_symbols[-2-curr_delay])))]
                    print(self.combine_symbols)

        self.NVAR.fit(**kwargs,X_train=X_train,Y_train=Y_train,Input_symbols=Input_symbols,Combine_symbols=self.combine_symbols)
        return

    def predict(self,Initial_data: np.array([]), Predict_time = 1, **kwargs):

        if Initial_data.shape[0] < self.delay:
            return
        if self.dim != Initial_data.shape[1]:
            return


        initialisation = Initial_data.shape[0]

        #Make extended data with the initial data, and add the already known delays:
        y_test = delay_data(np.append(Initial_data,np.zeros((Predict_time,self.dim)),axis=0),self.delay)

        for i in range(Predict_time):
            diff = self.NVAR.run_on_vector(y_test[initialisation + i - 1,:],**kwargs)
            current_result = y_test[initialisation + i - 1,:self.dim]  + diff

            #Put current result in correct positions:
            for shift in range(min(self.delay,Predict_time-i-1)+1):
                y_test[ (initialisation + i) + shift,self.dim * shift: self.dim * (shift+1)] = current_result

        return y_test[initialisation:,:self.dim]

    def get_symbolic_prediction(self):
        data_out_of_symbols = Data_Manipulation.data_out_of_symbols(delay = self.delay,dimension=self.dim)
        return self.predict(data_out_of_symbols)

