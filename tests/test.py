import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import unittest
import numpy as np
from core.Variable import Variable
from core.Function import add, square, exp
from core.Diff import numerical_diff


x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

print(x)
print(y)

z = add(square(x), square(y))

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