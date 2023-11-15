import matplotlib.pyplot as plt
import numpy as np
from typing import Union
from scipy.integrate import solve_ivp
def lorenz(xyz, *, s=10, r=28, b=2.667):

    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


def lorenzfull(
    n_timesteps: int = 10000,
    h: float = 0.01,                    #delta time
    rho: float = 28.0,
    sigma: float = 10.0,
    beta: float = 8.0 / 3.0,
    x0: Union[list, np.ndarray] = [0., 1., 1.05],       #starting conditions
    **kwargs,               #parameters passed to the scipy solve_ivp
) -> np.ndarray:


    def lorenz_diff(t, state):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    t_max = n_timesteps * h

    t_eval = np.linspace(0.0, t_max, n_timesteps)

    sol = solve_ivp(
        lorenz_diff, y0=x0, t_span=(0.0, t_max), t_eval=t_eval, **kwargs
    )

    return sol.y.T






xyzs = lorenzfull()

"""
# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
"""