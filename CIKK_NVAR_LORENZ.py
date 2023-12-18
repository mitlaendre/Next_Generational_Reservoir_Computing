# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays for Lorenz forecasting.  Don't be efficient for now.

@author: Dan


Edit: out_test initialization from zeroes() to ones()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import Data_Manipulation

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

# discrete-time versions of the times defined above
warmup_pts = round(warmup / dt)
traintime_pts = round(traintime / dt)
warmtrain_pts = warmup_pts + traintime_pts
testtime_pts = round(testtime / dt)
maxtime_pts = round(maxtime / dt)
plottime_pts = round(plottime / dt)
lyaptime_pts = round(lyaptime / dt)


print("warmup_pts")
print(warmup_pts)
print("traintime_pts")
print(traintime_pts)
print("testtime_pts")
print(testtime_pts)
print("plottime_pts")
print(plottime_pts)

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
ridge_param = 2.5e-6

# t values for whole evaluation time
# (need maxtime_pts + 1 to ensure a step of dt)
t_eval = np.linspace(0, maxtime, maxtime_pts + 1)

##
## Lorenz '63
##

sigma = 10
beta = 8 / 3
rho = 28


def lorenz(t, y):
    dy0 = sigma * (y[1] - y[0])
    dy1 = y[0] * (rho - y[2]) - y[1]
    dy2 = y[0] * y[1] - beta * y[2]

    # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
    return [dy0, dy1, dy2]


# I integrated out to t=50 to find points on the attractor, then use these as the initial conditions

lorenz_soln = solve_ivp(lorenz, (0, maxtime), [17.67715816276679, 12.931379185960404, 43.91404334248268], t_eval=t_eval,method='RK23')

# total variance of Lorenz solution
total_var = np.var(lorenz_soln.y[0:d, :])

##
## NVAR
##

# create an array to hold the linear part of the feature vector
x = np.zeros((dlin, maxtime_pts))

# fill in the linear part of the feature vector for all times
for delay in range(k):
    for j in range(delay, maxtime_pts):
        x[d * delay:d * (delay + 1), j] = lorenz_soln.y[:, j - delay]

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

# ridge regression: train W_out to map out_train to Lorenz[t] - Lorenz[t - 1]

W_out = (x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1:warmtrain_pts - 1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:, :] @ out_train[:, :].T + ridge_param * np.identity(dtot))
print("x_train.shape:")
print(out_train[:, :].shape)
print("x_train: ")
print(out_train[:, :])
print("y_train.shape: ")
print((x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1:warmtrain_pts - 1]).shape)
print("y_train: ")
print((x[0:d, warmup_pts:warmtrain_pts] - x[0:d, warmup_pts - 1:warmtrain_pts - 1]))

print("Fitted: ")
print(W_out)
# apply W_out to the training feature vector to get the training output
x_predict = x[0:d, warmup_pts - 1:warmtrain_pts - 1] + W_out @ out_train[:, 0:traintime_pts]


# create a place to store feature vectors for prediction
out_test = np.ones(dtot)  # full feature vector
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
    if j ==0:
        print("Prediction step:")
        print(x_test[0:d, j])
        print("+")
        print(W_out)
        print("@")
        print(out_test[:])

##
## Plot
##

t_linewidth = 1.1
a_linewidth = 0.3
plt.rcParams.update({'font.size': 12})

fig1 = plt.figure()
fig1.set_figheight(8)
fig1.set_figwidth(12)

xlabel = [10, 15, 20, 25, 30]
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

# true Lorenz attractor
axs1.plot(x[0, warmtrain_pts:maxtime_pts], x[2, warmtrain_pts:maxtime_pts], linewidth=a_linewidth)
axs1.set_xlabel('x')
axs1.set_ylabel('z')
axs1.set_title('ground truth')
axs1.text(-.25, .92, 'a)', ha='left', va='bottom', transform=axs1.transAxes)
axs1.axes.set_xbound(-21, 21)
axs1.axes.set_ybound(2, 48)

# training phase x
axs2.set_title('training phase')
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
axs6.set_title('testing phase')
axs6.set_xticks(xlabel)
axs6.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup,
          x[0, warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1], linewidth=t_linewidth)
axs6.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[0, 0:plottime_pts],
          linewidth=t_linewidth, color='r')
axs6.set_ylabel('x')
axs6.text(-.155, 0.87, 'f)', ha='left', va='bottom', transform=axs6.transAxes)
axs6.axes.xaxis.set_ticklabels([])
axs6.axes.set_xbound(9.7, 30.3)

# testing phase y
axs7.set_xticks(xlabel)
axs7.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup,
          x[1, warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1], linewidth=t_linewidth)
axs7.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[1, 0:plottime_pts],
          linewidth=t_linewidth, color='r')
axs7.set_ylabel('y')
axs7.text(-.155, 0.87, 'g)', ha='left', va='bottom', transform=axs7.transAxes)
axs7.axes.xaxis.set_ticklabels([])
axs7.axes.set_xbound(9.7, 30.3)

# testing phase z
axs8.set_xticks(xlabel)
axs8.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup,
          x[2, warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1], linewidth=t_linewidth)
axs8.plot(t_eval[warmtrain_pts - 1:warmtrain_pts + plottime_pts - 1] - warmup, x_test[2, 0:plottime_pts],
          linewidth=t_linewidth, color='r')
axs8.set_ylabel('z')
axs8.text(-.155, 0.87, 'h)', ha='left', va='bottom', transform=axs8.transAxes)
axs8.set_xlabel('time')
axs8.axes.set_xbound(9.7, 30.3)

plt.show()


print("Ground_truth.shape:")
print(x[:3, warmtrain_pts:warmtrain_pts+800].T.shape)
print("Ground_truth:")
print(x[:3, warmtrain_pts:warmtrain_pts+800].T)
print("Prediction.shape: ")
print(x_test[:3, :800].T.shape)
print("Prediction: ")
print(x_test[:3, :800].T)

Data_Manipulation.compare_3dData_2dPlot(x[:3, warmtrain_pts:warmtrain_pts+800].T,x_test[:3, :800].T)
Data_Manipulation.compare_3dData_3dPlot(x[:3, warmtrain_pts:warmtrain_pts+800].T,x_test[:3, :800].T)

"""OUTPUT:

warmup_pts
200
traintime_pts
400
testtime_pts
4800
plottime_pts
800
x_train.shape:
(28, 400)
x_train: 
[[  1.           1.           1.         ...   1.           1.
    1.        ]
 [ -3.0585347   -3.46482955  -3.98744    ...   3.77547281   4.61155107
    5.6416387 ]
 [ -4.4721737   -5.31544082  -6.32685765 ...   6.77314572   8.32126585
   10.18503806]
 ...
 [ 14.20615263  20.00033764  28.25391114 ...  30.39882639  45.87550295
   69.24346527]
 [-65.28237976 -73.75406705 -84.056539   ...  64.10165304  77.10998869
   94.97842498]
 [299.9960101  271.97852879 250.07163481 ... 135.17041314 129.61057586
  130.27801507]]
y_train.shape: 
(3, 400)
y_train: 
[[-0.40629485 -0.52261046 -0.65299045 ...  0.83607826  1.03008763
   1.25258677]
 [-0.84326712 -1.01141683 -1.20724358 ...  1.54812012  1.86377221
   2.18030332]
 [-0.67811812 -0.49804514 -0.2653101  ...  0.02927544  0.40872216
   0.94705144]]
Fitted: 
[[-4.00182949e-04  7.62657833e-01  3.22113871e-01 -2.21324377e-01
  -8.35840977e-01 -3.07696110e-01  2.07116998e-01  1.22130423e-01
  -2.63791066e-02 -2.17983842e-02 -1.89276810e-01 -3.29833714e-02
   6.15008950e-03  7.47845084e-03  1.45786531e-04  1.90911964e-02
   3.26644969e-03  1.06572078e-03  6.20978741e-03  1.63830414e-02
   3.73083033e-03  1.30837280e-04  7.33469748e-02  2.44013092e-02
  -1.70790116e-03 -3.65222461e-03 -1.79539685e-03 -5.55897111e-03]
 [ 1.88095346e-03  1.28704013e+00 -9.11691012e-02 -2.54791572e-01
  -4.61027239e-01  1.83200590e-03  2.38215634e-01  4.28075092e-01
  -9.07261126e-02  2.13174367e-01 -6.80103385e-01 -1.25364295e-01
  -2.88970062e-01  1.25923625e-02 -4.73428178e-02  6.95773111e-02
   1.41452108e-02  4.86944248e-02  8.74475829e-03 -1.88681285e-01
  -1.43576124e-02 -7.32458950e-05  2.70011467e-01  9.69737943e-02
   2.28548030e-01 -3.41114731e-05  2.16365564e-02 -7.58254988e-03]
 [ 3.78527911e-05  8.65419116e-03  5.81383070e-02 -4.50415537e-02
  -4.34334073e-02 -6.34131073e-02 -1.85154565e-02 -1.50710940e-01
  -2.38729894e-01  3.22038071e-02  1.84426546e-01  4.21440244e-01
  -2.89420193e-02  4.12583575e-02 -3.63307745e-03  2.15281129e-01
  -2.63447240e-02  3.30601053e-03 -1.17936308e-02 -2.41703092e-02
  -3.62493943e-03  1.88385442e-02 -5.21882772e-02 -3.27775866e-01
   2.29612715e-02 -4.03277613e-02  3.38819699e-03 -7.28711329e-03]]
Prediction step:
[ 6.89422547 12.36534138 12.7697131 ]
+
[[-4.00182949e-04  7.62657833e-01  3.22113871e-01 -2.21324377e-01
  -8.35840977e-01 -3.07696110e-01  2.07116998e-01  1.22130423e-01
  -2.63791066e-02 -2.17983842e-02 -1.89276810e-01 -3.29833714e-02
   6.15008950e-03  7.47845084e-03  1.45786531e-04  1.90911964e-02
   3.26644969e-03  1.06572078e-03  6.20978741e-03  1.63830414e-02
   3.73083033e-03  1.30837280e-04  7.33469748e-02  2.44013092e-02
  -1.70790116e-03 -3.65222461e-03 -1.79539685e-03 -5.55897111e-03]
 [ 1.88095346e-03  1.28704013e+00 -9.11691012e-02 -2.54791572e-01
  -4.61027239e-01  1.83200590e-03  2.38215634e-01  4.28075092e-01
  -9.07261126e-02  2.13174367e-01 -6.80103385e-01 -1.25364295e-01
  -2.88970062e-01  1.25923625e-02 -4.73428178e-02  6.95773111e-02
   1.41452108e-02  4.86944248e-02  8.74475829e-03 -1.88681285e-01
  -1.43576124e-02 -7.32458950e-05  2.70011467e-01  9.69737943e-02
   2.28548030e-01 -3.41114731e-05  2.16365564e-02 -7.58254988e-03]
 [ 3.78527911e-05  8.65419116e-03  5.81383070e-02 -4.50415537e-02
  -4.34334073e-02 -6.34131073e-02 -1.85154565e-02 -1.50710940e-01
  -2.38729894e-01  3.22038071e-02  1.84426546e-01  4.21440244e-01
  -2.89420193e-02  4.12583575e-02 -3.63307745e-03  2.15281129e-01
  -2.63447240e-02  3.30601053e-03 -1.17936308e-02 -2.41703092e-02
  -3.62493943e-03  1.88385442e-02 -5.21882772e-02 -3.27775866e-01
   2.29612715e-02 -4.03277613e-02  3.38819699e-03 -7.28711329e-03]]
@
[  1.           6.89422547  12.36534138  12.7697131    5.6416387
  10.18503806  11.82266166  47.53034487  85.2494515   88.03728133
  38.89472924  70.21794883  81.5080952  152.90166737 157.90186176
  69.76078847 125.94147254 146.19124745 163.06557264  72.04210762
 130.06001393 150.97199751  31.82808723  57.46030489  66.69918559
 103.73500027 120.41425901 139.77532881]
Ground_truth.shape:
(800, 3)
Ground_truth:
[[  8.38033209  14.77338672  14.45736693]
 [ 10.07996071  17.2008067   17.11221066]
 [ 11.90358527  19.22689009  20.88854732]
 ...
 [-11.182964    -7.66633285  34.0426804 ]
 [-10.19707044  -5.92809114  33.5947427 ]
 [ -9.08711587  -4.5564993   32.64070865]]
Prediction.shape: 
(800, 3)
Prediction: 
[[ 6.89422547 12.36534138 12.7697131 ]
 [ 8.38220174 14.77469448 14.45941565]
 [10.08113373 17.1993322  17.11327344]
 ...
 [-2.98857042 -1.45966353 23.59636401]
 [-2.68511854 -1.78114151 22.18550973]
 [-2.52581712 -2.15323257 20.87854572]]


"""