import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from core import Variable
from core import Function

x = Variable.Variable(np.array(0.5))

A = Function.Square()
B = Function.Exp()
C = Function.Square()

a = A(x)
b = B(a)
y = C(b)

print(y.data)
