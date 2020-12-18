import core
import numpy as np


def f(x):
    A = core.Function.Square()
    B = core.Function.Exp()
    C = core.Function.Square()
    return C(B(A(x)))


x = core.Variable.Variable(np.array(0.5))
dy = core.Diff.numerical_diff(f, x)

print(dy)
