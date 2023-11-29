import itertools as it
from scipy.special import comb
import numpy as np


def make_all_combinations(input = np.array([]),order = 1):
    if order <= 0:
        return np.array([1])
    return np.product(np.array(list(it.combinations(input,order)),dtype=float),1)

def create_combined_data( data, order=2, delay = 1):


    #make the delayed parts
    if len(data.shape) == 1:
        data_delayed = np.zeros(delay * data.shape[0])
        for delay in range(delay):
                data_delayed[data.shape[0] * delay:data.shape[0] * (delay + 1)] = data[:]
        out_data = np.ones((data.shape[0]*(delay+1)+1),dtype=float)
        out_data[1:] = data_delayed[:]

    else:
        data_delayed = np.zeros((delay*data.shape[0], data.shape[1]))
        for delay in range(delay):
            for i in range(delay, data.shape[1]):
                data_delayed[data.shape[0] * delay:data.shape[0] * (delay + 1), i] = data[:, i - delay]
        out_data = np.ones((data.shape[0]*(delay+1)+1, data.shape[1]),dtype=float)
        out_data[1:, :] = data_delayed[:,:]

    #append with combinated parts
    for i in range(2, order+1):
        out_data = np.append(out_data, make_all_combinations(data_delayed, i),axis=0)

    return out_data

class NVAR():

    """
    dim
    dim_lin
    dim_nonlin
    dim_total
    delay
    ridge
    order


    """



    def __init__(self, delay: int, order: int,strides: int, ridge: float):
        self.delay = delay
        self.order = order
        self.ridge = ridge


        return




    def fit(self,x_train, y_train, warmup=100):
        x_train = x_train.transpose()
        y_train = y_train.transpose()

        train_time = x_train.shape[1] - warmup


        out_train = create_combined_data(x_train,order = self.order, delay=self.delay)[:,warmup:]
        self.dim = x_train.shape[0]
        self.dim_total = out_train.shape[0]
        self.dim_lin = self.delay * self.dim
        self.dim_nonlin = self.dim_total - 1 - self.dim_lin



        self.W_out = (y_train[:, warmup:] - y_train[:,warmup-1:-1]) @ out_train[:, :].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T + self.ridge * np.identity(self.dim_total))

        x_predict = out_train[0+1:self.dim+1, :] + self.W_out @ out_train[:, 0:train_time]

        # calculate NRMSE between true Lorenz and training output
        rms = np.sqrt(np.mean((out_train[1:self.dim+1, :] - x_predict[:, :]) ** 2))
        print('Training nrmse: ' + str(rms))

        return

    def run_iterative(self,initial_vector = np.array([]), testtime = 1):

        y_test = np.zeros((self.dim_lin, testtime))  # linear part

        # copy over initial linear feature vector
        y_test[:, 0] = initial_vector


        for j in range(testtime - 1):

            # fill in the delay taps of the next state
            y_test[self.dim:self.dim_lin, j + 1] = y_test[0:(self.dim_lin - self.dim), j]

            # do a prediction
            out_test = create_combined_data(y_test[:, j], order=self.order, delay=self.delay)
            y_test[0:self.dim, j + 1] = y_test[0:self.dim, j] + self.W_out @ out_test[:]

        return y_test.transpose()



