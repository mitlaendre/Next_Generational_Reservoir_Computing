import numpy as np
import matplotlib.pyplot as plt
from Lorenz63 import lorenzfull

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

for changing in range(change_steps):
    if(howManySeeds!=0):
        for seedset in range(howManySeeds):
            set_seed(seedset*17)

            x = lorenzfull( length_train+length_test , 0.01)

            x_train = x[:length_train]
            y_train = x[1:length_train+1]
            x_test = x[length_train:-1]
            y_test = x[length_train+1:]


            res1 = Reservoir((changing+1)*1000, lr=0.9, sr=0.5)
            #res2 = Reservoir(100, sr=0.9, lr=0.3)

            readout = Ridge(ridge=1e-6)

            #deep_esn = (res1 >> res2)  >> readout
            deep_esn = (res1) >> readout

            predictions = deep_esn.fit(x_train,y_train).run(x_test)
            errors[changing,seedset] = errorFuncMSE(y_test,predictions)
            if(plotEverySeed):
                ax = plt.figure().add_subplot(projection='3d')
                ax.plot(*x.T, lw=0.5)
                ax.set_xlabel("X Axis")
                ax.set_ylabel("Y Axis")
                ax.set_zlabel("Z Axis")
                ax.set_title("Lorenz Attractor")

                ax = plt.figure().add_subplot(projection='3d')
                ax.plot(*predictions.T, lw=0.5)
                ax.set_xlabel("X Axis")
                ax.set_ylabel("Y Axis")
                ax.set_zlabel("Z Axis")
                ax.set_title("Predicted Lorenz Attractor")

                plt.show()
                plt.plot(x[:length_test])
                plt.show()

                plt.plot(predictions, label="Predictions")
                plt.plot(y_test, label="Ground truth",linestyle = "--")
                plt.legend()
                plt.show()


plt.plot(sum(errors.transpose()), label="Changing")
plt.legend()
plt.show()
