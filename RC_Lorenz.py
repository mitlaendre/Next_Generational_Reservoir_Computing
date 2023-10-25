import numpy as np
import matplotlib.pyplot as plt
from Lorenz63 import lorenzfull

from reservoirpy.nodes import Ridge, Reservoir, NVAR
from reservoirpy import set_seed

from joblib import Parallel, delayed


def generate_data(length_train, length_test):
    x = lorenzfull(length_train + length_test, 0.01)
    x_train = x[:length_train]
    y_train = x[1:length_train + 1]
    x_test = x[length_train:-1]
    y_test = x[length_train + 1:]
    return x_train, y_train, x_test, y_test


def create_esn(reservoir_size, reservoir_lr=0.9, reservoir_sr=0.5, ridge_reg=1e-6, ridge_name="default_ridge"):
    res = Reservoir(reservoir_size, lr=reservoir_lr, sr=reservoir_sr)
    readout = Ridge(ridge=ridge_reg, name=ridge_name)
    deep_esn = res >> readout
    return deep_esn


def error_func_mse(x, y):
    return np.average(np.power(sum(np.square(x - y)), 0.5))


def plot_data(x):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*x.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

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

def compare_3d_data(ground_truth, prediction):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*ground_truth.T, lw=0.5, label="ground truth")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    ax.plot(*prediction.T, lw=0.5, label="prediction")

    plt.legend()
    plt.show()


def experiment(data, reservoir_size, reservoir_lr=0.3, reservoir_sr=1.25, ridge_reg=1e-6, seed=0, warmup=100):
    x_train, y_train, x_test, y_test = data
    set_seed(seed)
    my_esn = create_esn(reservoir_size, reservoir_lr=reservoir_lr, reservoir_sr=reservoir_sr, ridge_reg=ridge_reg, ridge_name=str(reservoir_size) + str(seed) + str(reservoir_lr) + str(reservoir_sr))
    my_esn.fit(x_train, y_train, warmup=warmup)
    predictions = my_esn.run(x_test)
    error = error_func_mse(y_test, predictions)
    return error

def array_min_finder(input_array = np.array([0],dtype=object),maxthreads = 1): #output is a np.array with first element is the location vector, second is the minimum value
    if (input_array.ndim == 1):
        return np.array([np.array([np.argmin(input_array)]),np.amin(input_array)],dtype=object)
    else:
        this_dimensions_size = input_array.shape[0]
        sub_solutions = Parallel(n_jobs=min(input_array.size,maxthreads))(delayed(array_min_finder)(input_array[i],maxthreads = max(1,(maxthreads//input_array.size))) for i in range(this_dimensions_size))

        locations = np.full(this_dimensions_size,0,dtype=object)
        minimums = np.full(this_dimensions_size,0)
        for i in range(this_dimensions_size):
            locations[i] = sub_solutions[i][0]
            minimums[i] = sub_solutions[i][1]

        return np.array([np.append(np.array([np.argmin(minimums)]),locations[np.argmin(minimums)]),minimums[np.argmin(minimums)]],dtype=object)

def experiment_multiple(
        data,
        reservoir_size = np.array([100,1000,2000,5000]),
        reservoir_lr = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
        reservoir_sr = np.array([0.5,0.7,0.9,1,1.1,1.3,1.5,1.7]),
        ridge_reg=1e-6,
        seed = np.array([0,10,20,30,40,50,60,70,80,90]),
        warmup: int = 100,
        threads: int = 10):
    errors = Parallel(n_jobs=threads)(delayed(experiment)(data = data,
                                                          reservoir_size = reservoir_size[current_res_size].item(),
                                                          seed = seed[current_seed].item(),
                                                          reservoir_lr = reservoir_lr[current_reservoir_lr].item(),
                                                          reservoir_sr = reservoir_sr[current_reservoir_sr].item()
                                                          )
                                                            for current_seed in range(seed.size)
                                                            for current_res_size in range(reservoir_size.size)
                                                            for current_reservoir_lr in range(reservoir_lr.size)
                                                            for current_reservoir_sr in range(reservoir_sr.size)
                                                            )
    errors = np.array(errors).reshape(seed.size,reservoir_size.size,reservoir_lr.size,reservoir_sr.size)
    return errors


#multiple futi
x_train, y_train, x_test, y_test = generate_data(3000, 1000)
data = (x_train, y_train, x_test, y_test)




Reservoir_sizes = np.array([100,1000,2000])
Leaking_Rates = np.linspace(start = 0.1, stop = 0.9, num= 5)
Spectral_Radiuses = np.linspace(start = 0.5, stop = 1.5, num= 5)
Seeds = np.array([0,10,20])
#14 err 1000,0.3,0.5

"""
Reservoir_sizes = np.array([800,900,1000,1100,1200])
Leaking_Rates = np.linspace(start = 0.1, stop = 0.9, num= 8)
Spectral_Radiuses = np.linspace(start = 0.5, stop = 1.5, num= 8)
Seeds = np.array([10,20,30,40,50,60,70,80])
#17 err  1200,0.1,0.5

Reservoir_sizes = np.array([1100,1200,1300,1400,1500])
Leaking_Rates = np.linspace(start = 0.02, stop = 0.3, num= 8)
Spectral_Radiuses = np.linspace(start = 0.1, stop = 0.7, num= 8)
Seeds = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
#12 err 1200,0.1399,0.35714

Reservoir_sizes = np.array([1160,1180,1200,1220,1240])
Leaking_Rates = np.linspace(start = 0.1, stop = 0.2, num= 8)
Spectral_Radiuses = np.linspace(start = 0.15, stop = 0.45, num= 8)
Seeds = np.array([10,20,30,40,50,60,70,80,90,100,110,120,130,140,150])
#12 err 1180,0.1143,0.3643

"""

errors = experiment_multiple(data,reservoir_size=Reservoir_sizes,reservoir_lr=Leaking_Rates,reservoir_sr=Spectral_Radiuses,seed=Seeds,threads=4)
errors1 = sum(errors)/Seeds.size
best_setup = array_min_finder(errors1,maxthreads=4)
print(errors1)
print(best_setup[0])
print(best_setup[1])
print("Best reservoir size:")
print(Reservoir_sizes[best_setup[0][0]])
print("Best leaking rate:")
print(Leaking_Rates[best_setup[0][1]])
print("Best spectral radius:")
print(Spectral_Radiuses[best_setup[0][2]])

plot_errors_surface(errors1,Leaking_Rates=Leaking_Rates,Reservoir_sizes=Reservoir_sizes,Spectral_Radiuses=Spectral_Radiuses)

"""
#sima futi
x_train, y_train, x_test, y_test = generate_data(3000, 1000)
data = (x_train, y_train, x_test, y_test)
experiment((x_train, y_train, x_test, y_test), 300, seed=0)
"""


"""errors = np.full((num_res_sizes, num_seeds),0.)

for res_size in range(num_res_sizes):
    for seed in range(num_seeds):
        errors[res_size, seed] = experiment(data, (res_size + 1) * 500, seed=seed * 17)

plt.plot(sum(errors.transpose()), label="Error")
plt.legend()
plt.show()"""

"""my_esn = create_esn(3000, reservoir_lr=0.3, reservoir_sr=1.25)

my_esn.fit(x_train, y_train, warmup=100)

predictions = my_esn.run(x_test)

error = error_func_mse(y_test, predictions)

compare_3d_data(y_test, predictions)"""

"""for res_size in range(5):
    my_esn = create_esn((res_size + 5) * 100, reservoir_lr=0.3, reservoir_sr=1.25)
    my_esn.fit(x_train, y_train, warmup=100)
    predictions = my_esn.run(x_test)
    error = error_func_mse(y_test, predictions)
    compare_3d_data(y_test, predictions)
    print(error)"""