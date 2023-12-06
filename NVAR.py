import itertools as it
import numpy as np


def make_all_combinations(input = np.array([]),order = 1):

    if order <= 0:
        return np.full((1,input.shape[1]),1)
    if order > input.shape[0]:
        return np.array()
    return np.product(np.array(list(it.combinations(input,order)),dtype=float),1)


def combine_data( data, order=2):

    #Initialise with the constant and first order parts:
    out_data = np.ones((1, data.shape[1]), dtype=float)
    out_data = np.append(out_data,data[:, :],axis=0)

    #Append with combinated parts
    for i in range(2, order+1):
        out_data = np.append(out_data, make_all_combinations(data, i),axis=0)
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
        combined_x_train = combine_data(x_train,order = self.order)[:,:]

        #Fit the W_out matrix:
        self.W_out = self.y_train @ combined_x_train.T @ np.linalg.pinv(combined_x_train @ combined_x_train.T + self.ridge * np.identity(combined_x_train.shape[0]))
        return


    def run(self,x_data = np.array([])):

        #Make combinations:
        combined_x_data = combine_data(x_data,order = self.order)[:,:]

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[0],x_data.shape[1]))

        #Predict:
        y_data[:,:] = self.W_out @ combined_x_data[:]

        return y_data

    def runpoint(self,datapoint_vector):
        return self.run(np.array([datapoint_vector]).transpose())[:,0]