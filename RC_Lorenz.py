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
    my_esn = create_esn(reservoir_size, reservoir_lr=reservoir_lr, reservoir_sr=reservoir_sr, ridge_reg=ridge_reg, ridge_name=str(reservoir_size) + str(seed))
    my_esn.fit(x_train, y_train, warmup=warmup)
    predictions = my_esn.run(x_test)
    error = error_func_mse(y_test, predictions)
    return error

set_seed(0)


x_train, y_train, x_test, y_test = generate_data(3000, 1000)

data = (x_train, y_train, x_test, y_test)

num_res_sizes = 30;
num_seeds = 20;

"""errors = np.full((num_res_sizes, num_seeds),0.)

for res_size in range(num_res_sizes):
    for seed in range(num_seeds):
        errors[res_size, seed] = experiment(data, (res_size + 1) * 500, seed=seed * 17)

plt.plot(sum(errors.transpose()), label="Error")
plt.legend()
plt.show()"""

Parallel(n_jobs=10)(delayed(experiment)(data, (res_size + 1) * 500, seed=seed * 17) for res_size in range(num_res_sizes) for seed in range(num_seeds))

#experiment((x_train, y_train, x_test, y_test), 300)


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