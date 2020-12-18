import core
import numpy as np

A = core.Function.Square()
B = core.Function.Exp()
C = core.Function.Square()

x = core.Variable.Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# calculate gradient of y using backward propagation
y.grad = np.array(1.0)    # dy/dy
b.grad = C.backward(y.grad)    # dy/db
a.grad = B.backward(b.grad)    # dy/da
x.grad = A.backward(a.grad)    # dy/dx

print(x.grad)