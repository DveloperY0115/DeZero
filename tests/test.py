import numpy as np
from DeZero.core import *

x = Variable(np.array(2.0))

print(x ** 3)

y1 = 2.0 - x    # 0.0
y2 = x - 1.0    # 1.0
print(y1)
print(y2)

y1.backward()
print(x.grad)

x.cleargrad()

y2.backward()
print(x.grad)

"""
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
"""