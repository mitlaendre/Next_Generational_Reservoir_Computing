import itertools as it
import numpy as np
import sympy
x1 = sympy.symbols("x1")
x2 = sympy.symbols("x2")
x3 = sympy.symbols("x3")

a = np.array([x1,x2,x3])
print(a)
mtx = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(mtx)

res = mtx @ a.T
print(res)