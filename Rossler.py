import numpy as np
from typing import Union
from scipy.integrate import solve_ivp
def rossler(xyz, *, a = 0.2, b = 0.2, c = 5.7):

    x, y, z = xyz
    x_dot = -y -z
    y_dot = x + a*y
    z_dot = b + z*(x-c)
    return np.array([x_dot, y_dot, z_dot])

def rossler_diff(a,b,c,t, state):
    x, y, z = state
    return -y-z, x+a*y, b+z*(x-c)


def rosslerdata(
    n_timepoints: int = 4000,
    h: float = 0.01,                    #delta time
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
    x0: Union[list, np.ndarray] = [0., 1., 1.05],       #starting conditions
    **kwargs,               #parameters passed to the scipy solve_ivp
) -> np.ndarray:

    def rossler_diff_params(t,state): return rossler_diff(a,b,c,t,state)
    t_max = (n_timepoints) * h

    t_eval = np.linspace(0.0, t_max, n_timepoints+1)

    sol = solve_ivp(
        rossler_diff_params, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
    )

    return sol.y.T[:-1]


def rosslerdata(
        maxtime: float = 200,
        h: float = 0.01,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        x0: Union[list, np.ndarray] = [0., 1., 1.05],  # starting conditions
        **kwargs,  # parameters passed to the scipy solve_ivp
) -> np.ndarray:
    n_points = int(maxtime/h)
    t_eval = np.linspace(0, maxtime, n_points + 1)
    def rossler_diff_params(t, state): return rossler_diff(a,b,c, t, state)

    sol = solve_ivp(rossler_diff_params, (0, maxtime),x0, t_eval=t_eval,method='RK23')

    return sol.y.T[:-1]