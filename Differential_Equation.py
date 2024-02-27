import numpy as np
from typing import Union

import scipy.integrate
from scipy.integrate import solve_ivp

def Implemented_Steps(right_side,tn,Yn,dt,method = ""):

    if method == "Euler":
        fx = np.array(right_side(tn, Yn))
        return Yn + dt * fx
        # Yn+1 = Yn + dt * f(tn,Yn)


    elif method == "Midpoint":
        fx = np.array(right_side(tn + dt / 2., Yn + dt / 2. * np.array(right_side(tn, Yn))))
        return Yn + dt * fx
        # explicit: Yn+1 = Yn + dt * fx(tn + dt/2, Yn + dt/2 * fx(tn,Yn))
        # implicit: Yn+1 = Yn + dt * fx(tn + dt/2, 1/2 * (Yn + Yn+1))


    elif method == "RK4":
        k1 = np.array(right_side(tn, Yn))
        k2 = np.array(right_side(tn + dt/2,   Yn + dt/2 * k1))
        k3 = np.array(right_side(tn + dt/2,   Yn + dt/2 * k2))
        k4 = np.array(right_side(tn + dt,     Yn + dt   * k3))
        return Yn + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        # Yn+1 = Yn + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        # k1 = fx(tn,Yn)
        # k2 = fx(tn + h/2,Yn + h/2 * k1)
        # k3 = fx(tn + h/2,Yn + h/2 * k2)
        # k4 = fx(tn + h,Yn + h * k3)


    return


class Differential_Equation():
    def __init__(self, function = None):
        self.right_side = function
        return

    def generate_data(self,
            x0: Union[list, np.ndarray] = None,             # starting conditions
            maxtime: float = None,                          # max time of the generation    (optional)
            n_timepoints: int = None,                       # number of timepoints          (optional)
            dt: float = 0.01,                               # delta time
            method = ""                                     # method of the solution
    ):
        if n_timepoints == None:
            n_timepoints = int(maxtime / dt + 1)
        else:
            maxtime = (n_timepoints - 1) * dt
        t_eval = np.linspace(0.0, maxtime, n_timepoints)

        if method in {"","RK45","RK23","DOP853","Radau","BDF","LSODA"}: #Solve_Ivp implemented methods
            sol = solve_ivp(
                self.right_side, y0=x0, t_span=(0.0, maxtime), t_eval=t_eval, method = method)
            sol = sol.y
        else:
            x0 = np.array(x0)
            sol = np.full((x0.shape[0], t_eval.shape[0]), 0.)
            sol[:, 0] = x0
            for i, t in enumerate(t_eval):
                if i > 0:
                    sol[:, i] = Implemented_Steps(right_side=self.right_side,tn=t,Yn=sol[:,i-1],dt=dt,method=method)
        print("data generated:")
        print(sol.T[:-1])
        return sol.T[:-1]
class Lorenz63(Differential_Equation):

    def __init__(self, rho: float = 28.0, sigma: float = 10.0, beta: float = 8.0 / 3.0):
        def fx(t, state):
            x, y, z = state
            return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
        super().__init__(fx)
        return

class Rossler(Differential_Equation):

    def __init__(self, a: float = 0.2, b: float = 0.2, c: float = 5.7):
        def fx(t, state):
            x, y, z = state
            return -y - z, x + a * y, b + z * (x - c)
        super().__init__(fx)
        return

class Chua(Differential_Equation):

    def default_function(x: float):
        return x**3 /16 - (x  /6)

    def __init__(self, a: float = 9.267, b: float = 14., f_function = default_function ):
        def fx(t, state):
            x, y, z = state
            return a*(y-f_function(x)), x - y + z, -b*y
        super().__init__(fx)
        return

