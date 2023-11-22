import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib.pyplot
import reservoirpy

#np.array használata
"""
teszt1 = np.full((10,2),0)
teszt2 = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(teszt1)
print("")
print(teszt2)"""

"""
#Paralell és egy nagy tömb kezelése

from joblib import Parallel, delayed
def process(i,j,k,l):
    return i*1000 + j*100 + k*10 + l

results = Parallel(n_jobs=5)(delayed(process)(i,j,k,l) for i in range(2) for j in range(3) for k in range(4) for l in range(5))
print(results)
print("==================================================================")
results1 = np.array(results).reshape(2,3,4,5)
print(results1)
print("==================================================================")
results2 = sum(results1) #i =0,1 adta össze
print(results2)
print("==================================================================")
print(results2[2,3,4]) #i-re osszeadva, j = 2, k = 3, l = 4 elem kiirat
"""


#idő kezelése
""" 
import time

start = time.time()

print(23*2.3)

end = time.time()
print(end - start)
"""


#félkész array min index kereső
"""
x = np.array([[3,2,3],[4,5,6],[7,1,9]])
k=np.argmin(x)

print(k)

a = x
ment = np.full(x.ndim,0)
for i in range(x.ndim):
    ment[i] = k/a.size
    k %= a.size
    a = a[0]

print(k)
print(ment)
"""


#rekurzív parralel array min index kereső
"""
from joblib import Parallel, delayed
def array_min_finder(input_array = np.array([0],dtype=object)): #output is a np.array with first element is the location vector, second is the minimum value
    #print("run with input:")
    #print(input_array)
    if (input_array.ndim == 1):
        return np.array([np.array([np.argmin(input_array)]),np.amin(input_array)],dtype=object)
    else:
        this_dimensions_size = input_array.shape[0]
        sub_solutions = Parallel(n_jobs=input_array.size)(delayed(array_min_finder)(input_array[i]) for i in range(this_dimensions_size))

        locations = np.full(this_dimensions_size,0,dtype=object)
        minimums = np.full(this_dimensions_size,0)
        for i in range(this_dimensions_size):
            locations[i] = sub_solutions[i][0]
            minimums[i] = sub_solutions[i][1]

        return np.array([np.append(np.array([np.argmin(minimums)]),locations[np.argmin(minimums)]),minimums[np.argmin(minimums)]],dtype=object)
"""



#surface plot for errors
"""
tomb = np.array(
                [[[ 58.15923757,  64.4747462,   90.53759092, 113.50645819, 168.68428256],
                  [ 90.97613044,  84.64348166,  82.05740897,  80.47934262,  81.86836333],
                  [112.70567593, 107.54013518, 105.62494151,  98.99207874,  90.34202357],
                  [125.74188876, 122.31270676, 118.16988091, 112.9041992,  108.98378016],
                  [132.83822029, 130.95516304, 126.76384502, 122.85519774, 125.81939769]],

                 [[ 18.70495611,  28.05702499,  63.78762039, 106.34228975, 187.01847576],
                  [ 14.53528095,  23.32955401,  26.84213527,  37.16252606,  43.25204842],
                  [ 28.47863387,  33.24142741,  42.47417412,  72.54227207, 105.75431941],
                  [ 37.12749575,  49.97675714,  74.12889605, 136.41838343, 337.76633145],
                  [ 45.05699719,  47.62938513,  95.05599729, 251.87652286, 409.35448071]],

                 [[ 21.29363039,  27.17554758,  73.97252802, 150.40597075, 276.60478843],
                  [ 25.65613492,  27.16656747,  27.96123024,  36.99924734,  43.84407999],
                  [ 27.27305612,  36.18156575,  55.57702243,  57.5072704,   75.53505627],
                  [ 33.20185763,  40.53210217,  75.3829493, 136.42583232, 143.50443268],
                  [ 38.7618769,   48.31108057,  88.69132182, 239.97998492, 341.20780761]]])

Reservoir_sizes = np.array([100,1000,2000])
Leaking_Rates = np.linspace(start = 0.1, stop = 0.9, num= 5)
Spectral_Radiuses = np.linspace(start = 0.5, stop = 1.5, num= 5)
Seeds = np.array([0,10,20])

def plot_errors_surface(input_errors = np.array([]),Reservoir_sizes = np.array([]),Leaking_Rates = np.array([]),Spectral_Radiuses = np.array([])):
    for i in range(input_errors.shape[0]):
        hf = plt.pyplot.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(Leaking_Rates, Spectral_Radiuses)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_surface(X, Y, input_errors[i])
        ha.set_xlabel('$Leaking Rate$')
        ha.set_ylabel('$Spectral Radius$')
        ha.set_zlabel(r'$Average error$')
        plt.pyplot.show()


plot_errors_surface(tomb,Leaking_Rates = Leaking_Rates,Spectral_Radiuses=Spectral_Radiuses,Reservoir_sizes=Reservoir_sizes)"""

#help for parametric plots
"""
x = np.array([[1,2],[1.1,1.9],[1.2,1.8],[1.15,1.85],[1.1,1.9],[1,2]])
plt.plot(np.transpose(x)[0],np.transpose(x)[1])
plt.show()"""


import numpy as np
import matplotlib.pyplot as plt
import Lorenz63

plotting = True


##
## Parameters
##

# time step
dt = 0.025
# units of time to warm up NVAR. need to have warmup_pts >= 1
warmup = 5.
# units of time to train for
traintime = 10.
# units of time to test for
testtime = 120.
# total time to run for
maxtime = warmup + traintime + testtime
# how much of testtime to plot
plottime = 20.
# Lyapunov time of the Lorenz system
lyaptime = 1.104

# discrete-time versions of the times defined
warmup_pts = round(warmup / dt)
traintime_pts = round(traintime / dt)
warmtrain_pts = warmup_pts + traintime_pts
testtime_pts = round(testtime / dt)
maxtime_pts = round(maxtime / dt)
plottime_pts = round(plottime / dt)
lyaptime_pts = round(lyaptime / dt)

# input dimension
d = 3
# number of time delay taps
k = 2
# size of linear part of feature vector
dlin = k * d
# size of nonlinear part of feature vector
dnonlin = int(dlin * (dlin + 1) / 2)
# total size of feature vector: constant + linear + nonlinear
dtot = 1 + dlin + dnonlin

# ridge parameter for regression
ridge_param = 1.4e-2

# t values for whole evaluation time
# (need maxtime_pts + 1 to ensure a step of dt)
t_eval = np.linspace(0, maxtime, maxtime_pts + 1)

#making the groundtruth
lorenz_y = Lorenz63.lorenzfull(n_timesteps= int(maxtime/dt),h = dt, x0=  [17.67715816276679, 12.931379185960404, 43.91404334248268]).transpose()


##
## NVAR
##

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin, maxtime_pts))

# fill in the linear part of the feature vector for all times
for delay in range(k):
    for j in range(delay, maxtime_pts):
        x[d * delay:d * (delay + 1), j] = lorenz_y[:, j - delay]

# create an array to hold the full feature vector for training time
# (use ones so the constant term is already 1)
out_train = np.ones((dtot, traintime_pts))

# copy over the linear part (shift over by one to account for constant)
out_train[1:dlin + 1, :] = x[:, warmup_pts - 1:warmtrain_pts - 1]

# fill in the non-linear part
cnt = 0
for row in range(dlin):
    for column in range(row, dlin):
        # shift by one for constant
        out_train[dlin + 1 + cnt] = x[row, warmup_pts - 1:warmtrain_pts - 1] * x[column,
                                                                               warmup_pts - 1:warmtrain_pts - 1]
        cnt += 1





#===============================Training

# ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]
W_out = (x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1:warmtrain_pts - 1]) @ out_train[:,
                                                                                        :].T @ np.linalg.pinv(
    out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(dtot))

# apply W_out to the training feature vector to get the training output
x_predict = x[0:d, warmup_pts - 1:warmtrain_pts - 1] + W_out @ out_train[:, 0:traintime_pts]

# calculate NRMSE between true Lorenz and training output
rms = np.sqrt(np.mean((x[0:d, warmup_pts:warmtrain_pts] - x_predict[:, :]) ** 2))
print('training nrmse: ' + str(rms))

# create a place to store feature vectors for prediction
out_test = np.zeros(dtot)  # full feature vector
x_test = np.zeros((dlin, testtime_pts))  # linear part

# copy over initial linear feature vector
x_test[:, 0] = x[:, warmtrain_pts - 1]






# do prediction
for j in range(testtime_pts - 1):
    # copy linear part into whole feature vector
    out_test[1:dlin + 1] = x_test[:, j]  # shift by one for constant
    # fill in the non-linear part
    cnt = 0
    for row in range(dlin):
        for column in range(row, dlin):
            # shift by one for constant
            out_test[dlin + 1 + cnt] = x_test[row, j] * x_test[column, j]
            cnt += 1
    # fill in the delay taps of the next state
    x_test[d:dlin, j + 1] = x_test[0:(dlin - d), j]
    # do a prediction
    x_test[0:d, j + 1] = x_test[0:d, j] + W_out @ out_test[:]





if plotting:

    t_linewidth = 1.1
    a_linewidth = 0.3
    plt.rcParams.update({'font.size': 12})

    fig1 = plt.figure()
    fig1.set_figheight(8)
    fig1.set_figwidth(12)

    xlabel = [10, 15, 20, 25, 30, 35, 40]
    h = 120
    w = 100

    # top left of grid is 0,0
    axs1 = plt.subplot2grid(shape=(h, w), loc=(0, 9), colspan=22, rowspan=38)
    axs2 = plt.subplot2grid(shape=(h, w), loc=(52, 0), colspan=42, rowspan=20)
    axs3 = plt.subplot2grid(shape=(h, w), loc=(75, 0), colspan=42, rowspan=20)
    axs4 = plt.subplot2grid(shape=(h, w), loc=(98, 0), colspan=42, rowspan=20)
    axs5 = plt.subplot2grid(shape=(h, w), loc=(0, 61), colspan=22, rowspan=38)
    axs6 = plt.subplot2grid(shape=(h, w), loc=(52, 50), colspan=42, rowspan=20)
    axs7 = plt.subplot2grid(shape=(h, w), loc=(75, 50), colspan=42, rowspan=20)
    axs8 = plt.subplot2grid(shape=(h, w), loc=(98, 50), colspan=42, rowspan=20)

    # true NOISY Lorenz attractor
    axs1.plot(x[0, warmtrain_pts:maxtime_pts], x[2, warmtrain_pts:maxtime_pts], linewidth=a_linewidth)
    axs1.set_xlabel('x')
    axs1.set_ylabel('z')
    axs1.set_title('Ground truth')
    axs1.text(-.25, .92, 'a)', ha='left', va='bottom', transform=axs1.transAxes)
    axs1.axes.set_xbound(-21, 21)
    axs1.axes.set_ybound(2, 48)

    # training phase x
    axs2.set_title('Training phase')
    axs2.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x[0, warmup_pts:warmtrain_pts], linewidth=t_linewidth)
    axs2.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x_predict[0, :], linewidth=t_linewidth, color='r')
    axs2.set_ylabel('x')
    axs2.text(-.155, 0.87, 'b)', ha='left', va='bottom', transform=axs2.transAxes)
    axs2.axes.xaxis.set_ticklabels([])
    axs2.axes.set_ybound(-21., 21.)
    axs2.axes.set_xbound(-.15, 10.15)

    # training phase y
    axs3.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x[1, warmup_pts:warmtrain_pts], linewidth=t_linewidth)
    axs3.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x_predict[1, :], linewidth=t_linewidth, color='r')
    axs3.set_ylabel('y')
    axs3.text(-.155, 0.87, 'c)', ha='left', va='bottom', transform=axs3.transAxes)
    axs3.axes.xaxis.set_ticklabels([])
    axs3.axes.set_xbound(-.15, 10.15)

    # training phase z
    axs4.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x[2, warmup_pts:warmtrain_pts], linewidth=t_linewidth)
    axs4.plot(t_eval[warmup_pts:warmtrain_pts] - warmup, x_predict[2, :], linewidth=t_linewidth, color='r')
    axs4.set_ylabel('z')
    axs4.text(-.155, 0.87, 'd)', ha='left', va='bottom', transform=axs4.transAxes)
    axs4.set_xlabel('time')
    axs4.axes.set_xbound(-.15, 10.15)

    # prediction attractor
    axs5.plot(x_test[0, :], x_test[2, :], linewidth=a_linewidth, color='r')
    axs5.set_xlabel('x')
    axs5.set_ylabel('z')
    axs5.set_title('NG-RC prediction')
    axs5.text(-.25, 0.92, 'e)', ha='left', va='bottom', transform=axs5.transAxes)
    axs5.axes.set_xbound(-21, 21)
    axs5.axes.set_ybound(2, 48)

    # testing phase x
    axs6.set_title('Testing phase')
    axs6.set_xticks(xlabel)
    axs6.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, lorenz_y[0, 0:plottime_pts],
              linewidth=t_linewidth)
    axs6.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[0, 0:plottime_pts],
              linewidth=t_linewidth, color='r')
    axs6.set_ylabel('x')
    axs6.text(-.155, 0.87, 'f)', ha='left', va='bottom', transform=axs6.transAxes)
    axs6.axes.xaxis.set_ticklabels([])
    axs6.axes.set_xbound(9.7, 30.3)

    # testing phase y
    axs7.set_xticks(xlabel)
    axs7.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, lorenz_y[1, 0:plottime_pts],
              linewidth=t_linewidth)
    axs7.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[1, 0:plottime_pts],
              linewidth=t_linewidth, color='r')
    axs7.set_ylabel('y')
    axs7.text(-.155, 0.87, 'g)', ha='left', va='bottom', transform=axs7.transAxes)
    axs7.axes.xaxis.set_ticklabels([])
    axs7.axes.set_xbound(9.7, 30.3)

    # testing phase z
    axs8.set_xticks(xlabel)
    axs8.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, lorenz_y[2, 0:plottime_pts],
              linewidth=t_linewidth)
    axs8.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[2, 0:plottime_pts],
              linewidth=t_linewidth, color='r')
    axs8.set_ylabel('z')
    axs8.text(-.155, 0.87, 'h)', ha='left', va='bottom', transform=axs8.transAxes)
    axs8.set_xlabel('time')
    axs8.axes.set_xbound(9.7, 30.3)

    plt.show()


