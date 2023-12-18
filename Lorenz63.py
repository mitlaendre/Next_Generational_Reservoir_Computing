import numpy as np
from typing import Union
from scipy.integrate import solve_ivp
def lorenz(xyz, *, s=10, r=28, b=2.667):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def lorenz_diff(sigma,rho,beta,t, state):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


def lorenzdata(
    n_timepoints: int = 4000,
    h: float = 0.01,                    #delta time
    rho: float = 28.0,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    x0: Union[list, np.ndarray] = [0., 1., 1.05],       #starting conditions
    **kwargs,               #parameters passed to the scipy solve_ivp
) -> np.ndarray:

    def lorenz_diff_params(t,state): return lorenz_diff(sigma,rho,beta,t,state)
    t_max = (n_timepoints) * h

    t_eval = np.linspace(0.0, t_max, n_timepoints+1)

    sol = solve_ivp(
        lorenz_diff_params, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
    )

    return sol.y.T[:-1]


def lorenzdata(
        maxtime: float = 200,
        h: float = 0.01,
        rho: float = 28.0,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        x0: Union[list, np.ndarray] = [0., 1., 1.05],  # starting conditions
        **kwargs,  # parameters passed to the scipy solve_ivp
) -> np.ndarray:
    n_points = int(maxtime/h)
    t_eval = np.linspace(0, maxtime, n_points + 1)
    def lorenz_diff_params(t, state): return lorenz_diff(sigma, rho, beta, t, state)

    sol = solve_ivp(lorenz_diff_params, (0, maxtime),x0, t_eval=t_eval,method='RK23')

    return sol.y.T[:-1]