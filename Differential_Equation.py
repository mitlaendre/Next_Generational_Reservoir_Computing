import numpy as np
from typing import Union
from scipy.integrate import solve_ivp
class Differential_Equation():
    def __init__(self, function = None):
        self.right_side = function
        return

    def generate_data(self,
            maxtime: float = 10,                              # max time of the generation    (optional)
            n_timepoints: int = None,                       # number of timepoints          (optional)
            dt: float = 0.01,                               # delta time
            x0: Union[list, np.ndarray] = [0., 0., 0.],     # starting conditions
            **kwargs                                        # parameters passed to the scipy solve_ivp
    ):

        if n_timepoints == None: n_timepoints = int(maxtime / dt + 1)
        else: maxtime = (n_timepoints-1) * dt

        t_eval = np.linspace(0.0, maxtime, n_timepoints)
        sol = solve_ivp(
            self.right_side, y0=x0, t_span=(0.0, maxtime), t_eval=t_eval, **kwargs
        )

        return sol.y.T[:-1]

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

    def __init__(self, a: float = 1., b: float = 1., f_function = default_function ):
        def fx(t, state):
            x, y, z = state
            return a*(y-f_function(x)), x - y + z, -b*y
        super().__init__(fx)
        return