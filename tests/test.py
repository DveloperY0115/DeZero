import numpy as np
import core.core as core

x = core.Variable(np.array(2.0))
y = core.Variable(np.array(3.0))

z = x * y

print(z)

z.backward()

"""
x = Variable(np.array(2.0))
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)
"""