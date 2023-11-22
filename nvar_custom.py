import itertools as it
from scipy.special import comb
import numpy as np
class NVAR():

    def __init__(self, delay: int, order: int,strides: int, ridge: float):
        self.delay = delay
        self.order = order
        self.ridge = ridge


        return
    def fit(self,x_train, y_train, warmup=100):
        x_train = x_train.transpose()
        # discrete-time versions of the times defined
        warmup_pts = warmup
        traintime_pts = x_train.shape[1] - warmup_pts
        warmtrain_pts = x_train.shape[1]

        # ridge parameter for regression
        ridge_param = self.ridge


        d = x_train.shape[0]

        self.d = d
        # number of time delay taps
        k = self.delay
        # size of linear part of feature vector
        self.dlin = k * d
        # size of nonlinear part of feature vector
        dnonlin = int(self.dlin * (self.dlin + 1) / 2)
        # total size of feature vector: constant + linear + nonlinear
        self.dtot = 1 + self.dlin + dnonlin

        x_train_all = np.zeros((self.dlin, warmtrain_pts))

        # fill in the linear part of the feature vector for all times
        for delay in range(k):
            for j in range(delay, warmtrain_pts):
                x_train_all[d * delay:d * (delay + 1), j] = x_train[:, j - delay]

        # create an array to hold the full feature vector for training time
        # (use ones so the constant term is already 1)
        out_train = np.ones((self.dtot, traintime_pts))

        # copy over the linear part (shift over by one to account for constant)
        out_train[1:self.dlin + 1, :] = x_train_all[:, warmup_pts - 1:warmtrain_pts - 1]

        # fill in the non-linear part
        cnt = 0
        for row in range(self.dlin):
            for column in range(row, self.dlin):
                # shift by one for constant
                out_train[self.dlin + 1 + cnt] = x_train_all[row, warmup_pts - 1:warmtrain_pts - 1] * x_train_all[column,
                                                                                       warmup_pts - 1:warmtrain_pts - 1]
                cnt += 1

        # ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]
        self.W_out = (x_train_all[0:d, warmup_pts:warmtrain_pts] - x_train_all[0:d, warmup_pts - 1:warmtrain_pts - 1]) @ out_train[:,
                                                                                                :].T @ np.linalg.pinv(
            out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(self.dtot))

        # apply W_out to the training feature vector to get the training output
        x_predict = x_train_all[0:d, warmup_pts - 1:warmtrain_pts - 1] + self.W_out @ out_train[:, 0:traintime_pts]

        # calculate NRMSE between true Lorenz and training output
        rms = np.sqrt(np.mean((x_train_all[0:d, warmup_pts:warmtrain_pts] - x_predict[:, :]) ** 2))
        print('training nrmse: ' + str(rms))

        self.initial_feature_vector = x_train_all[:, warmtrain_pts - 1]

        return

    def run(self,x_test):

        testtime_pts = x_test.shape[0]

        # create a place to store feature vectors for prediction
        out_test = np.zeros(self.dtot)  # full feature vector
        x_test = np.zeros((self.dlin, testtime_pts))  # linear part

        # copy over initial linear feature vector
        x_test[:, 0] = self.initial_feature_vector


        #out_test = np.full(self.dlin+1,0)
        for j in range(testtime_pts - 1):
            # copy linear part into whole feature vector
            out_test[1:self.dlin + 1] = x_test[:, j]  # shift by one for constant
            # fill in the non-linear part
            cnt = 0
            for row in range(self.dlin):
                for column in range(row, self.dlin):
                    # shift by one for constant
                    out_test[self.dlin + 1 + cnt] = x_test[row, j] * x_test[column, j]
                    cnt += 1
            # fill in the delay taps of the next state
            x_test[self.d:self.dlin, j + 1] = x_test[0:(self.dlin - self.d), j]
            # do a prediction
            x_test[0:self.d, j + 1] = x_test[0:self.d, j] + self.W_out @ out_test[:]

        return x_test.transpose()



