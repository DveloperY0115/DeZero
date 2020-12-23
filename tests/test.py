import numpy as np
from core.core import *

x = Variable(np.array(2.0))

# y = x + np.array(3.0)
y = np.array([3.0]) + x

z = x + 3.0
z1 = 3.0 + x

print(y)
print(z)
print(z1)

"""
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
"""