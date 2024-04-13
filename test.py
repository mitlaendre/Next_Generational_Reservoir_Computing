import Differential_Equation
import NVAR
import Data_Manipulation
import sympy
import numpy as np
from scipy.integrate import solve_ivp


dif = Differential_Equation.Chua()

x,y,z = sympy.symbols('x y z')

sol = solve_ivp(dif.right_side, y0=[0.1,0.1,0.1], t_span=(0.0, 0.025), t_eval=[0.0,0.025], method = "Radau")
print(sol.y)

sol = solve_ivp(dif.right_side, y0=[x,y,z], t_span=(0.0, 0.025), t_eval=[0.0,0.025], method = "Radau")
print(sol.y)

