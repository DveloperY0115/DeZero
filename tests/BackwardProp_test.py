import core
import numpy as np

from core.Function import square, exp

x = core.Variable.Variable(np.array(0.5))
y = square(exp(square(x)))
y.backward()
print(x.grad)
