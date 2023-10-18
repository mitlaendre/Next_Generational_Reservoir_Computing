import numpy as np
import matplotlib.pyplot as plt
from Lorenz63 import lorenzfull
from joblib import Parallel, delayed

from reservoirpy.nodes import Ridge, Reservoir, NVAR
from reservoirpy import set_seed


howManySeeds: int = 3;
change_steps: int = 5;

plotEverySeed: bool = False;
length_train = 2000;
length_test = 500;

errors = np.full((change_steps,howManySeeds),0.)



def errorFuncMSE(Original,Prediction):
    return np.average(np.power(sum(np.square(Original-Prediction)),0.5))




def RC_Lorenc_run(lorenz_array, seed = np.array(0), RC_units = 100, RC_lr = 1, RC_sr = 1):

    x = lorenz_array


    x_train = x[:length_train]
    y_train = x[1:length_train+1]
    x_test = x[length_train:-1]
    y_test = x[length_train+1:]


    res1 = Reservoir(units = RC_units, lr = RC_lr, sr = RC_sr)

    readout = Ridge(ridge=1e-6)
    deep_esn = (res1) >> readout
    predictions = deep_esn.fit(x_train,y_train).run(x_test)
    return errorFuncMSE(y_test,predictions)



array = lorenzfull( length_train+length_test , 0.01)
lr = 1
sr = 1


results = Parallel(n_jobs=2)(delayed(RC_Lorenc_run)(array,seed*17,(units+1)*100,lr,sr) for seed in range(howManySeeds) for units in range(change_steps))
results1 = np.array(results).reshape(-1, 10)

plt.plot(sum(results1.transpose()), label="Changing")
plt.legend()
plt.show()
