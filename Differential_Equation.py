import numpy as np
from typing import Union

import scipy.integrate
from scipy.integrate import solve_ivp

import Data_Manipulation


def Implemented_Steps(right_side,tn,Yn,dt,method):

    if method == "Euler":
        fx = np.array(right_side(tn, Yn))
        return Yn + dt * fx
    elif method == "Midpoint":
        fx = np.array(right_side(tn + dt / 2., Yn + dt / 2. * np.array(right_side(tn, Yn))))
        return Yn + dt * fx
    else:
        if method != "RK4":
            print("Numerical method not found, doing RK4 ")
        k1 = np.array(right_side(tn, Yn))
        k2 = np.array(right_side(tn + dt/2,   Yn + dt/2 * k1))
        k3 = np.array(right_side(tn + dt/2,   Yn + dt/2 * k2))
        k4 = np.array(right_side(tn + dt,     Yn + dt   * k3))
        return Yn + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def Implemented_Multisteps(right_side,tn,dt,method,last_datapoints):
    #last_datapoints contains: [...,Yn-2,Yn-1,Yn] and we calculate Yn+1
    try:
        if method == "Adams-Bashforth 1":
            fx = np.array(right_side(tn, last_datapoints[:,-1]))
            return last_datapoints[:,-1] + dt * fx
        elif method == "Adams-Bashforth 2":
            fx = (1.5*np.array(right_side(tn,last_datapoints[:,-1]))-0.5*np.array(right_side(tn-dt,last_datapoints[:,-2])))
            return last_datapoints[:,-1] + dt*fx
        elif method == "Adams-Bashforth 3":
            fx = (23. * np.array(right_side(tn,last_datapoints[:,-1])) - 16.*np.array(right_side(tn-dt,last_datapoints[:,-2])) + 5.*np.array(right_side(tn-2*dt, last_datapoints[:,-3])))/12.
            return last_datapoints[:,-1] + dt * fx
        elif method == "Adams-Bashforth 4":
            fx = (55.*np.array(right_side(tn,last_datapoints[:,-1])) - 59.*np.array(right_side(tn-dt,last_datapoints[:,-2])) + 37.*np.array(right_side(tn-2*dt,last_datapoints[:,-2])) - 9.*np.array(right_side(tn-3*dt,last_datapoints[:,-3])))/24.
            return last_datapoints[:,-1] + dt * fx
        elif method == "Adams-Bashforth 5":
            fx = (1901.*np.array(right_side(tn,last_datapoints[:,-1])) - 2774.*np.array(right_side(tn-dt,last_datapoints[:,-2])) + 2616.*np.array(right_side(tn-2*dt,last_datapoints[:,-3])) - 1274.*np.array(right_side(tn-3*dt,last_datapoints[:,-4])) + 251.*np.array(right_side(tn-4*dt,last_datapoints[:,-5])))/720.
            return last_datapoints[:,-1] + dt * fx

    except IndexError: #If there is not enough datapoints yet for multistep, then do RK4:
        return Implemented_Steps(right_side = right_side,tn = tn,dt = dt,Yn = last_datapoints[:,-1],method = "RK4")


def W_out_generator(input_symbols, combine_symbols, combination_vector,dt):
    W_out = np.zeros((len(input_symbols), len(combine_symbols)))
    for i in range(W_out.shape[0]):
        for j in range(W_out.shape[1]):
            W_out[i, j] = combination_vector[i].coeff(combine_symbols[j])*dt
    return W_out


class Differential_Equation():
    def __init__(self, function = None):
        self.right_side = function
        self.dt = 0.
        self.symbolic_equation = None
        return

    def generate_data(self,
            x0: np.ndarray = None,             # starting conditions
            maxtime: float = None,                          # max time of the generation    (optional)
            n_timepoints: int = None,                       # number of timepoints          (optional)
            dt: float = 0.01,                               # delta time
            method = "",                                     # method of the solution
            equation_symbols = [],
                      **kwargs
    ):
        x0 = np.array(x0)

        if len(x0.shape) == 1: x0 = np.array([x0])
        self.dt = dt
        if n_timepoints == None:
            n_timepoints = int(maxtime / dt + 1)
        else:
            maxtime = (n_timepoints - 1) * dt
        t_eval = np.linspace(0.0, maxtime, n_timepoints)

        if method in {"","RK45","RK23","DOP853","Radau","BDF","LSODA"}:                             #Solve_Ivp implemented methods
            symbolic_data = Data_Manipulation.data_out_of_symbols(delay=1,dimension=x0.shape[1], input_symbols=equation_symbols)[0]
            sol = solve_ivp(self.right_side, y0=x0[-1], t_span=(0.0, maxtime), t_eval=t_eval, method = method)
            sol = sol.y
            #self.equation_symbols = solve_ivp(self.right_side, y0=symbolic_data, t_span=(0.0, dt), t_eval=[0.0,dt], method = method)
            self.symbolic_equation = []


        elif method in {"Adams-Bashforth 1","Adams-Bashforth 2","Adams-Bashforth 3","Adams-Bashforth 4","Adams-Bashforth 5","Adams-Bashforth"}:             #Implemented multistep methods
            if method == "Adams-Bashforth": method = "Adams-Bashforth 5"
            sol = np.full((x0.shape[1], t_eval.shape[0] + x0.shape[0] - 1), 0.)
            sol[:, :x0.shape[0]] = x0[:].T
            for i, t in enumerate(t_eval):
                if i > 0:
                    sol[:, i] = Implemented_Multisteps(right_side=self.right_side, tn=t, dt=dt, method=method,last_datapoints=sol[:,:i])
            #make one symbolic step too
            symbolic_data = Data_Manipulation.data_out_of_symbols(delay=10, dimension=x0.shape[1], input_symbols=equation_symbols).T
            self.symbolic_equation = Implemented_Multisteps(right_side=self.right_side, tn=0., dt=dt, method=method,last_datapoints=symbolic_data)

        else:                                                                                                       #Implemented single step methods
            sol = np.full((x0.shape[1], t_eval.shape[0]), 0.)
            sol[:, 0] = x0[-1]
            for i, t in enumerate(t_eval):
                if i > 0:
                    sol[:, i] = Implemented_Steps(right_side=self.right_side,tn=t,Yn=sol[:,i-1],dt=dt,method=method)
            #Make one symbolic step too
            symbolic_data = Data_Manipulation.data_out_of_symbols(delay=1, dimension=x0.shape[1],input_symbols=equation_symbols)[0]
            self.symbolic_equation = Implemented_Steps(right_side=self.right_side, tn=0., dt=dt, method=method,Yn=symbolic_data)
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

