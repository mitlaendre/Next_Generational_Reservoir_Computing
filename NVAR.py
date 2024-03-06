import itertools as it
import numpy as np
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


class NVAR():

    def __init__(self, order: int, ridge: float):

        self.y_train = None             #y_train data
        self.x_train = None             #x_train data
        self.combined_x_train = None    #x_train with all the combinations and the constant
        self.W_out = None               #trained W_out matrix
        self.order = order              #order of the combinations
        self.ridge = ridge              #ridge parameter
        self.norm_data = False
        self.cutoff_small_weights = 0.
        return


    def fit(self, x_train: np.array([]), y_train: np.array([]), norm_data = False, cutoff_small_weights = 0.) -> None:
        if norm_data:
            self.norm_data = True
            self.x_deviation = x_train.std(0)
            self.x_mean = x_train.mean(0)
            self.y_deviation = y_train.std(0)
            self.y_mean = y_train.mean(0)

            self.x_train = (x_train-self.x_mean)/self.x_deviation
            self.y_train = (y_train - self.y_mean) / self.y_deviation
        else:
            self.x_train = x_train
            self.y_train = y_train
        self.cutoff_small_weights = cutoff_small_weights

        #Make combinations:
        self.combined_x_train = combine_data(self.x_train,order = self.order)

        #Fit the W_out matrix:
        self.W_out = self.y_train.T @ self.combined_x_train @ np.linalg.pinv(self.combined_x_train.T @ self.combined_x_train + self.ridge * np.identity(self.combined_x_train.shape[1]))

        if self.cutoff_small_weights != 0.:
            for i in range(self.W_out.shape[0]):
                for j in range(self.W_out.shape[1]):
                    if np.abs(self.W_out[i,j]) <= self.cutoff_small_weights:
                        self.W_out[i,j] = 0.

        return


    def run(self,x_data = np.array([])):
        if(self.W_out.any == None):
            print("Error: The NVAR has not been trained yet!")
            return
        if x_data.shape[1] != self.x_train.shape[1]:
            print("The data dimension is not matching the training data")
            return

        if self.norm_data:
            x_data1 = (x_data-self.x_mean)/self.x_deviation
        else:
            x_data1 = x_data

        #Make combinations:
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
        return self.run(np.array([datapoint_vector]))[:,0]

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
        return