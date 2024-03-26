import itertools as it
import numpy as np
import Data_Manipulation
import sympy

def make_all_combinations(input = np.array([]),order = 1):
    #cast touple, list and others to np.array([])
    if type(input) != type(np.array([])):
        input = np.array(input)

    if order <= 0:
        return np.full((input.shape[0],1),1)
    return np.prod(np.array(list(it.combinations_with_replacement(input.T,order))),1).T


def combine_data( in_data = np.array([[]]), order=2):

    if len(in_data.shape) == 1: #if vector, cast up to matrix, then back
        data = np.array([in_data])
    else:
        data = in_data

    #Initialise with the constant and first order parts:
    out_data = np.ones((data.shape[0],1), dtype=float)
    out_data = np.append(out_data,data[:, :],axis=1)

    #Append with combinated parts
    for i in range(2, order+1):
        out_data = np.append(out_data, make_all_combinations(data, i),axis=1)

    if len(in_data.shape) == 1: #if vector
        out_data = out_data[0]

    return out_data

def custom_combine_data(in_data = np.array([[]]),input_symbols = [],combine_symbols = []):
    if len(in_data.shape) == 1: #if vector, cast up to matrix, then back
        data = np.array([in_data])
    else:
        data = in_data
    print(data)
    if data.shape[1] != len(input_symbols):
        return

    out_data = np.full((data.shape[0], len(combine_symbols)), 0., dtype=object)
    for i in range(data.shape[0]):
        for j in range(len(combine_symbols)):
            if (type(combine_symbols[j]) == type(0.)) or (type(combine_symbols[j]) == type(0)):
                out_data[i,j] = combine_symbols[j]
            else:
                out_data[i, j] = combine_symbols[j].subs(dict(zip(input_symbols, data[i])))


    if len(in_data.shape) == 1:  # if vector
        out_data = out_data[0]

    return out_data

class NVAR():

    def __init__(self, order = 0, ridge = 0., dict = {}):
        if "NVAR" in dict:
            dict = dict["NVAR"]
        if "Order" not in dict:
            dict["Order"] = order
        if "Ridge" not in dict:
            dict["Ridge"] = ridge




        self.y_train = None             #y_train data
        self.x_train = None             #x_train data
        self.combined_x_train = None    #x_train with all the combinations and the constant
        self.W_out = None               #trained W_out matrix
        self.order = dict["Order"]              #order of the combinations
        self.ridge = dict["Ridge"]              #ridge parameter
        self.norm_data = None
        self.cutoff_small_weights = None
        self.custom_combinations = None

        return

    def initialize_symbols(self, combined_symbols = False):
        if combined_symbols:
            self.combine_symbols = combine_data(np.array([self.input_symbols]),order= self.order)[0]
        else:
            self.input_symbols = Data_Manipulation.data_out_of_symbols(delay=0, dimension=self.x_train.shape[1])[0]


    def fit(self, x_train = np.array([]), y_train = np.array([]), norm_data = False, cutoff_small_weights = 0.,input_symbols = [],combine_symbols = [], dict = {}) -> None:
        if "NVAR" in dict:
            dict = dict["NVAR"]
        if "X_train" not in dict:
            dict["X_train"] = x_train
        if "Y_train" not in dict:
            dict["Y_train"] = y_train
        if "Norm data" not in dict:
            dict["Norm data"] = norm_data
        if "Cutoff small weights" not in dict:
            dict["Cutoff small weights"] = cutoff_small_weights
        if "Input symbols" not in dict:
            dict["Input symbols"] = input_symbols
        if "Combine symbols" not in dict:
            dict["Combine symbols"] = combine_symbols


        if dict["Norm data"]:
            self.norm_data = True
            self.x_deviation = dict["X_train"].std(0)
            self.x_mean = dict["X_train"].mean(0)
            self.y_deviation = dict["Y_train"].std(0)
            self.y_mean = dict["Y_train"].mean(0)

            self.x_train = (dict["X_train"]-self.x_mean)/self.x_deviation
            self.y_train = (dict["Y_train"] - self.y_mean) / self.y_deviation
        else:
            self.x_train = dict["X_train"]
            self.y_train = dict["Y_train"]
        self.cutoff_small_weights = dict["Cutoff small weights"]

        if len(dict["Input symbols"]) == 0:
            self.initialize_symbols(combined_symbols=False)
            self.custom_combinations = False
        else:
            self.input_symbols = dict["Input symbols"]
            if self.custom_combinations == None:
                self.custom_combinations = True

        if len(dict["Combine symbols"]) == 0:
            self.initialize_symbols(combined_symbols=True)
            self.custom_combinations = False
        else:
            self.combine_symbols = dict["Combine symbols"]
            if self.custom_combinations == None:
                self.custom_combinations = True

        print("Input symbols:")
        print(self.input_symbols)
        print("Combine symbols:")
        print(self.combine_symbols)

        #Make combinations:
        if self.custom_combinations:
            self.combined_x_train = custom_combine_data(self.x_train,self.input_symbols,self.combine_symbols)
        else:
            self.combined_x_train = combine_data(self.x_train,order = self.order)

        print(self.combined_x_train)
        #Fit the W_out matrix:
        self.W_out = self.y_train.T @ self.combined_x_train @ np.linalg.pinv(self.combined_x_train.T @ self.combined_x_train + self.ridge * np.identity(self.combined_x_train.shape[1]))

        if self.cutoff_small_weights != 0.:
            for i in range(self.W_out.shape[0]):
                for j in range(self.W_out.shape[1]):
                    if np.abs(self.W_out[i,j]) <= self.cutoff_small_weights:
                        self.W_out[i,j] = 0.

        return


    def run(self,x_data = np.array([]), dict = {}):
        if "NVAR" in dict:
            dict = dict["NVAR"]
        if "X_data" not in dict:
            dict["X_data"] = x_data


        if(self.W_out.any == None):
            print("Error: The NVAR has not been trained yet!")
            return
        if dict["X_data"].shape[1] != self.x_train.shape[1]:
            print("The data dimension is not matching the training data")
            return

        if self.norm_data:
            x_data1 = (dict["X_data"]-self.x_mean)/self.x_deviation
        else:
            x_data1 = dict["X_data"]

        #Make combinations:
        if self.custom_combinations:
            combined_x_test = custom_combine_data(x_data1,self.input_symbols,self.combine_symbols)
        else:
            combined_x_test = combine_data(x_data1,order = self.order)

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[1],x_data1.shape[1]),dtype=x_data1.dtype)

        #Predict:
        y_data[:,:] = self.W_out @ combined_x_test.T

        if self.norm_data:
            y_data = (y_data.T*self.y_deviation+self.y_mean).T

        return y_data

    def run_on_vector(self,datapoint_vector):
        #wrapping the vector in data structure, then unwrap
        return self.run(dict = {"X_data" : np.array([datapoint_vector])})[:,0]

    def debug_print(self):
        print("Ridge param: ")
        print(self.ridge)
        print("Order: ")
        print(self.order)

        if not(self.W_out.any == None):
            print("X_train data: ")
            print(self.x_train)
            print("X_train shape: ")
            print(self.x_train.shape)
            print("Y_train data: ")
            print(self.y_train)
            print("Y_train shape: ")
            print(self.y_train.shape)
            print("Combined x_train data: ")
            print(self.combined_x_train)
            print("W_out matrix: ")
            print(self.W_out)
        else:
            print("NVAR not trained yet")

        print("input symbols:")
        print(self.input_symbols)
        print("combine symbols:")
        print(self.combine_symbols)
        print("Custom combination:")
        print(self.custom_combinations)
        return