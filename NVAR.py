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
    """
    self.x_train        x_train data
    self.y_train        y_train data
    self.W_out          trained W_out matrix
    self.order          order of the combinations
    self.ridge          ridge parameter
    """


    def __init__(self, order: int, ridge: float):

        self.x_train = None
        self.y_train = None
        self.W_out = None
        self.order = order
        self.ridge = ridge
        return


    def fit(self, x_train: np.array([]), y_train: np.array([])) -> None:

        self.x_train = x_train
        self.y_train = y_train

        #Make combinations:
        combined_x_train = combine_data(self.x_train,order = self.order)

        #Fit the W_out matrix:
        self.W_out = self.y_train.T @ combined_x_train @ np.linalg.pinv(combined_x_train.T @ combined_x_train + self.ridge * np.identity(combined_x_train.shape[1]))
        print("x_train.shape")
        print(combined_x_train.T.shape)
        print("x_train: ")
        print(combined_x_train.T)
        print("y_train.shape")
        print(y_train.T.shape)
        print("y_train: ")
        print(self.y_train.T)

        print("Fitted: ")
        print(self.W_out)
        return


    def run(self,x_data = np.array([])):
        #Make combinations:
        combined_x_data = combine_data(x_data,order = self.order)

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[1],x_data.shape[1]))

        #Predict:
        y_data[:,:] = self.W_out @ combined_x_data.T

        return y_data

    def runpoint(self,datapoint_vector):
        return self.run(np.array([datapoint_vector]))[:,0]