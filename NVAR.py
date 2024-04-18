import itertools as it
import numpy as np
import Data_Manipulation
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

def make_all_combinations(input = np.array([]),order = 1):  #making all the combinations with the order given
    input = np.array(input)     #cast touple, list and others to np.array([])
    if order <= 0:
        return np.full((input.shape[0],1),1)
    return np.prod(np.array(list(it.combinations_with_replacement(input.T,order))),1).T

def norm(a: np.array,axis = 0,p=2):
    def vectornorm(a): return np.sum(a**p)**(1./p)
    return np.apply_along_axis(vectornorm,axis=axis,arr=a)

class NVAR():

    def combine_data(self,in_data,Initialise_combinations = False):
        in_data = np.array(in_data)
        if len(in_data.shape) == 1:     data = np.array([in_data])  # if it's a vector, cast up to matrix, then back
        else:                           data = in_data


        if Initialise_combinations:     # Initialise with the constant and first order parts:
            out_data = np.ones((data.shape[0], 1), dtype=float)
            out_data = np.append(out_data, data[:, :], axis=1)

            for i in range(2, self.order + 1):      # Append with combinated parts (according to saved order)
                out_data = np.append(out_data, make_all_combinations(data, i), axis=1)

        else:       #make the saved combinations
            if data.shape[1] != len(self.input_symbols):
                print("Data shape not matching symbolic shape")
                return

            out_data = np.full((data.shape[0], len(self.combine_symbols)), 0.)
            for timepoint in range(data.shape[0]):
                for combination_number in range(len(self.combine_symbols)):
                    if (type(self.combine_symbols[combination_number]) == type(0.)) or (type(self.combine_symbols[combination_number]) == type(0)):
                        out_data[timepoint, combination_number] = self.combine_symbols[combination_number]      #If just a number, then leave it
                    else:       #If symbolic, then calculate accordingly
                        out_data[timepoint, combination_number] = self.combine_symbols[combination_number].subs(dict(zip(self.input_symbols, data[timepoint])))


        if len(in_data.shape) == 1: out_data = out_data[0]  # if it was a vector, cast back
        return out_data
    def regression_fit(self):
        if self.lasso == 0.:
            #W = Y.T @ X (X.T @ X + Ridge)^-1
            self.W_out = self.y_train.T @ self.combined_x_train @ np.linalg.pinv(self.combined_x_train.T @ self.combined_x_train + self.ridge * np.identity(self.combined_x_train.shape[1]))
        else:
            Wridge = self.ridge
            Wlasso = self.lasso

            a = Wlasso
            b = Wridge * 2.
            alpha = a+b
            l1_ratio = a/(a+b)

            tol = 1e-4 #tolerance of the konvergence
            max_iter = 100000 #max iterations of the konvergence

            regr = ElasticNet(l1_ratio=l1_ratio,alpha=alpha,fit_intercept=False,tol=tol,max_iter=max_iter)
            regr.fit(self.combined_x_train, self.y_train)
            self.W_out = regr.coef_
    def __init__(self,Ridge : float,Lasso = 0., Order = 1 ,**kwargs):

        self.y_train = None             #y_train data
        self.x_train = None             #x_train data
        self.combined_x_train = None    #x_train with all the combinations and the constant

        self.input_symbols = None
        self.combine_symbols = None

        self.W_out = None               #trained W_out matrix
        self.order = Order            #order of the combinations

        self.ridge = Ridge            #ridge parameter
        self.lasso = Lasso

        self.norm_data = None
        self.cutoff_small_influences = None
        self.influences_history = []
        return


    def fit(self,X_train: np.array([]),Y_train: np.array([]),Input_symbols = [],Combine_symbols = [],Cutoff_small_influences = 0.,Norm_data = False,**kwargs) -> None:
        if "Ridge" in kwargs: self.ridge = kwargs["Ridge"]
        if "Lasso" in kwargs: self.lasso = kwargs["Lasso"]

        #Saving some variables
        self.norm_data = Norm_data
        self.cutoff_small_influences = Cutoff_small_influences

        #Norming data
        if self.norm_data:
            self.x_deviation, self.y_deviation = X_train.std(0), Y_train.std(0)
            self.x_mean, self.y_mean = X_train.mean(0), Y_train.mean(0)
            self.x_train, self.y_train = (X_train - self.x_mean) / self.x_deviation ,(Y_train - self.y_mean) / self.y_deviation
        else:
            self.x_train, self.y_train = X_train,Y_train

        #Making the symbolic fitting
        if len(Input_symbols) == 0:
            self.input_symbols = Data_Manipulation.data_out_of_symbols(dimension=self.x_train.shape[1])[0]
        else:   self.input_symbols = Input_symbols

        if len(Combine_symbols) == 0:
            self.combine_symbols = self.combine_data(Input_symbols,Initialise_combinations=True)
        else:   self.combine_symbols = Combine_symbols

        #Make combinations:
        self.combined_x_train = self.combine_data(self.x_train)

        #Fit the W_out matrix:
        self.regression_fit()

        if self.cutoff_small_influences != 0.:
            while (len(self.influences_history) == 0) or (np.min(self.influences_history[-1]) <= self.cutoff_small_influences):

                influences = np.zeros(self.combined_x_train.shape[1])
                for combinator in range(self.combined_x_train.shape[1]):
                    influences[combinator] = np.max(norm(np.array([self.W_out[:,combinator]]).T @ np.array([self.combined_x_train[:,combinator]]),axis=0) / norm(self.y_train,axis=1))
                self.influences_history.append(influences)
                if ("Printing" in kwargs) and ("Enable_printing" in kwargs["Printing"]) and kwargs["Printing"]["Enable_printing"]:
                    print("Overall influences of the combinators: ")
                    print(influences)
                    print("for the combinators: ")
                    print(self.combine_symbols)

                if np.min(influences) <= self.cutoff_small_influences:
                    min_index = np.argmin(influences)
                    self.combined_x_train = np.delete(self.combined_x_train,min_index,axis=1)
                    self.combine_symbols = np.delete(self.combine_symbols,min_index,axis=0)
                    self.regression_fit()

    def run(self,X_data: np.array([]),**kwargs):
        if(self.W_out.any == None):
            print("Error: The NVAR has not been trained yet!")
            return
        if X_data.shape[1] != self.x_train.shape[1]:
            print("The data dimension is not matching the training data")
            return

        if self.norm_data:
            X_data = (X_data-self.x_mean)/self.x_deviation

        #Make combinations:
        combined_x_test = self.combine_data(X_data)

        #Inicialise the result data:
        y_data = np.zeros((self.y_train.shape[1],X_data.shape[1]),dtype=X_data.dtype)

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