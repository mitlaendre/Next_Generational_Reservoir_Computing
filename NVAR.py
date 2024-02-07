import itertools as it
import numpy as np


def make_all_combinations(input = np.array([]),order = 1):

    if order <= 0:
        return np.full((input.shape[0],1),1)
    if order > input.shape[1]:
        return np.array()
    return np.product(np.array(list(it.combinations_with_replacement(input.T,order)),dtype=float),1).T


def combine_data( data, order=2):
    #Initialise with the constant and first order parts:
    out_data = np.ones((data.shape[0],1), dtype=float)
    out_data = np.append(out_data,data[:, :],axis=1)

    #Append with combinated parts
    for i in range(2, order+1):
        out_data = np.append(out_data, make_all_combinations(data, i),axis=1)
    return out_data


class NVAR():

    def __init__(self, order: int, ridge: float):

        self.y_train = None             #y_train data
        self.x_train = None             #x_train data
        self.combined_x_train = None    #x_train with all the combinations and the constant
        self.W_out = None               #trained W_out matrix
        self.order = order              #order of the combinations
        self.ridge = ridge              #ridge parameter
        return


    def fit(self, x_train: np.array([]), y_train: np.array([])) -> None:
        self.x_train = x_train
        self.y_train = y_train

        #Make combinations:
        self.combined_x_train = combine_data(self.x_train,order = self.order)

        #Fit the W_out matrix:
        self.W_out = self.y_train.T @ self.combined_x_train @ np.linalg.pinv(self.combined_x_train.T @ self.combined_x_train + self.ridge * np.identity(self.combined_x_train.shape[1]))

        return


    def run(self,x_data = np.array([])):
        if(self.W_out.any == None):
            print("Error: The NVAR has not been trained yet!")
            return

        #Make combinations:
        combined_x_test = combine_data(x_data,order = self.order)

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[1],x_data.shape[1]))

        #Predict:
        y_data[:,:] = self.W_out @ combined_x_test.T

        return y_data

    def run_vector(self,datapoint_vector):
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