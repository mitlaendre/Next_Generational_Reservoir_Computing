import numpy as np
import matplotlib.pyplot as plt
import Differential_Equation
from joblib import Parallel, delayed
import random

def plot_3dData_3dPlot(x):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*x.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

def plot_3dData_2dPlot(x):
    plt.plot(np.transpose(x)[0],"r-",legend = "x")
    plt.plot(np.transpose(x)[1],"g-",legend = "y")
    plt.plot(np.transpose(x)[2],"b-",legend = "z")
    plt.show()

    x1 = np.full((x.shape[0],2),0)
    for i in range(x1.shape[0]):
        x1[i,0] = x[i,0]+x[i,1]
        x1[i,1] = x[i,2]
    plt.plot(np.transpose(x1)[0],np.transpose(x1)[1])
    plt.show()

    return

def compare_3dData_3dPlot(ground_truth, prediction):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*ground_truth.T, lw=0.5, label="ground truth")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.plot(*prediction.T, lw=0.5, label="prediction")

    plt.legend()
    plt.show()
    return

def compare_3dData_2dPlot(ground_truth, prediction):
    plt.plot(np.transpose(ground_truth)[0], "r-", label="x")
    plt.plot(np.transpose(ground_truth)[1], "g-", label="y")
    plt.plot(np.transpose(ground_truth)[2], "b-", label="z")
    plt.plot(np.transpose(prediction)[0], "r--", label="Prediction x")
    plt.plot(np.transpose(prediction)[1], "g--", label="Prediciton y")
    plt.plot(np.transpose(prediction)[2], "b--", label="Prediciton z")
    plt.show()
    return

def plot_errors_surface(input_errors = np.array([]),Reservoir_sizes = np.array([]),Leaking_Rates = np.array([]),Spectral_Radiuses = np.array([])):
    for i in range(input_errors.shape[0]):
        hf = plt.figure()
        ha = hf.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(Leaking_Rates, Spectral_Radiuses)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_surface(X, Y, input_errors[i])
        ha.set_xlabel('$Leaking Rate$')
        ha.set_ylabel('$Spectral Radius$')
        ha.set_zlabel(r'$Average error$')
        plt.show()
        return


#unfinished, example run below
def bifurcate_plot(fix_param: float, n_skip: int, n_shown_iter: int, step: int = 1, param_interval_min: float = 0.0, param_interval_max: float = 0.1):
    interval = np.linspace(param_interval_min, param_interval_max, step)

    def func(atadott):
        diffegy = Differential_Equation.Chua(a=atadott, b=fix_param)
        kezdo = [random.randrange(-1,1),random.randrange(-1,1),random.randrange(-1,1)]
        data = diffegy.generate_data(n_timepoints=n_skip + n_shown_iter, dt=0.1, x0=kezdo)
        X = data[n_skip:, 0]
        Y = data[n_skip:, 1]
        Z = data[n_skip:, 2]
        A = np.full(n_shown_iter - 1, atadott)
        return (X,Y,Z,A)

    res = Parallel(n_jobs=20)(delayed(func)(atadott) for atadott in interval)

    As = np.array([])
    Xs = np.array([])
    Ys = np.array([])
    Zs = np.array([])
    for i in range(len(res)):
        As = np.append(As,res[i][3])
        Xs = np.append(Xs,res[i][0])
        Ys = np.append(Ys, res[i][1])
        Zs = np.append(Zs, res[i][2])

    dpi = 180
    plt.plot(As, Xs, ls='', marker=',', color='black')
    plt.xlim(param_interval_min, param_interval_max)
    plt.savefig('bifurcation_' + str(param_interval_min) + '-' + str(param_interval_max) + '_x_' + str(step) + '.png',dpi=dpi)
    plt.plot(As, Ys, ls='', marker=',', color='black')
    plt.xlim(param_interval_min, param_interval_max)
    plt.savefig('bifurcation_' + str(param_interval_min) + '-' + str(param_interval_max) + '_y_' + str(step) + '.png',dpi=dpi)
    plt.plot(As, Zs, ls='', marker=',', color='black')
    plt.xlim(param_interval_min, param_interval_max)
    plt.savefig('bifurcation_' + str(param_interval_min) + '-' + str(param_interval_max) + '_z_' + str(step) + '.png',dpi=dpi)

    return
#bifurcate_plot(seed = 14,n_skip= 100000,n_iter= 100, r_min=6, r_max=14,step=200)
