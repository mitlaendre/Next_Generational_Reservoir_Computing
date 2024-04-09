import itertools as it
import numpy as np
import Data_Manipulation

def make_all_combinations(input = np.array([]),order = 1):
    #cast touple, list and others to np.array([])
    if type(input) != type(np.array([])):
        input = np.array(input)

    if order <= 0:
        return np.full((input.shape[0],1),1)
    return np.prod(np.array(list(it.combinations_with_replacement(input.T,order))),1).T


class NVAR():

    def combine_data(self,in_data):
        if len(in_data.shape) == 1:  # if vector, cast up to matrix, then back
            data = np.array([in_data])
        else:
            data = in_data

        if self.custom_combinations:
            if data.shape[1] != len(self.input_symbols):
                print("Data shape not matching input symbols length")
                return

            out_data = np.full((data.shape[0], len(self.combine_symbols)), 0.)
            for i in range(data.shape[0]):
                for j in range(len(self.combine_symbols)):
                    if (type(self.combine_symbols[j]) == type(0.)) or (type(self.combine_symbols[j]) == type(0)):
                        out_data[i, j] = self.combine_symbols[j]
                    else:
                        out_data[i, j] = self.combine_symbols[j].subs(dict(zip(self.input_symbols, data[i])))

        else:
            # Initialise with the constant and first order parts:
            out_data = np.ones((data.shape[0], 1), dtype=float)
            out_data = np.append(out_data, data[:, :], axis=1)

            # Append with combinated parts
            for i in range(2, self.order + 1):
                out_data = np.append(out_data, make_all_combinations(data, i), axis=1)

        if len(in_data.shape) == 1:  # if vector
            out_data = out_data[0]

        return out_data


    def __init__(self,Order: int, Ridge: float,**kwargs):

        self.y_train = None             #y_train data
        self.x_train = None             #x_train data
        self.combined_x_train = None    #x_train with all the combinations and the constant
        self.W_out = None               #trained W_out matrix
        self.order = Order            #order of the combinations
        self.ridge = Ridge            #ridge parameter
        self.norm_data = None
        self.cutoff_small_weights = None
        self.custom_combinations = None

        return

    def initialize_symbols(self, combined_symbols = False):
        if combined_symbols:
            self.combine_symbols = self.combine_data(np.array([self.input_symbols]))[0]
        else:
            self.input_symbols = Data_Manipulation.data_out_of_symbols(dimension=self.x_train.shape[1])[0]


    def fit(self,X_train: np.array([]),Y_train: np.array([]),Input_symbols = [],Combine_symbols = [],Cutoff_small_weights = 0.,Norm_data = False,**kwargs) -> None:

        if Norm_data:
            self.norm_data = True
            self.x_deviation = X_train.std(0)
            self.x_mean = X_train.mean(0)
            self.y_deviation = Y_train.std(0)
            self.y_mean = Y_train.mean(0)

            self.x_train = (X_train-self.x_mean)/self.x_deviation
            self.y_train = (Y_train - self.y_mean) / self.y_deviation
        else:
            self.x_train = X_train
            self.y_train = Y_train
        self.cutoff_small_weights = Cutoff_small_weights

        if len(Input_symbols) == 0:
            self.initialize_symbols(combined_symbols=False)
            self.custom_combinations = False
        else:
            self.input_symbols = Input_symbols
            if self.custom_combinations == None:
                self.custom_combinations = True

        if len(Combine_symbols) == 0:
            self.initialize_symbols(combined_symbols=True)
            self.custom_combinations = False
        else:
            self.combine_symbols = Combine_symbols
            if self.custom_combinations == None:
                self.custom_combinations = True

        #Make combinations:
        self.combined_x_train = self.combine_data(self.x_train)

        #Fit the W_out matrix:
        self.W_out = self.y_train.T @ self.combined_x_train @ np.linalg.pinv(self.combined_x_train.T @ self.combined_x_train + self.ridge * np.identity(self.combined_x_train.shape[1]))

        if self.cutoff_small_weights != 0.:
            for i in range(self.W_out.shape[0]):
                for j in range(self.W_out.shape[1]):
                    if np.abs(self.W_out[i,j]) <= self.cutoff_small_weights:
                        self.W_out[i,j] = 0.
        return


    def run(self,X_data: np.array([]),**kwargs):

        if(self.W_out.any == None):
            print("Error: The NVAR has not been trained yet!")
            return
        if X_data.shape[1] != self.x_train.shape[1]:
            print("The data dimension is not matching the training data")
            return

        if self.norm_data:
            x_data1 = (X_data-self.x_mean)/self.x_deviation
        else:
            x_data1 = X_data

        #Make combinations:
        combined_x_test = self.combine_data(x_data1)

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[1],x_data1.shape[1]),dtype=x_data1.dtype)

        #Predict:
        y_data[:,:] = self.W_out @ combined_x_test.T

        if self.norm_data:
            y_data = (y_data.T*self.y_deviation+self.y_mean).T

        return y_data

    def run_on_vector(self,datapoint_vector,**kwargs):
        #wrapping the vector in data structure, then unwrap
        return self.run(**{"X_data" : np.array([datapoint_vector])},**kwargs)[:,0]

    def debug_print(self):
        print("Ridge param: ")
        print(self.ridge)
        print("Order: ")
        print(self.order)

        print("input symbols:")
        print(self.input_symbols)
        print("combine symbols:")
        print(self.combine_symbols)
        print("Custom combination:")
        print(self.custom_combinations)

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
        return