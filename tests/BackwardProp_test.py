import core
import numpy as np

A = core.Function.Square()
B = core.Function.Exp()
C = core.Function.Square()

x = core.Variable.Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

"""
# calculate gradient of y using backward propagation
y.grad = np.array(1.0)    # dy/dy
b.grad = C.backward(y.grad)    # dy/db
a.grad = B.backward(b.grad)    # dy/da
x.grad = A.backward(a.grad)    # dy/dx

print(x.grad)
"""

# automatically calculate gradients
y.grad = np.array(1.0)
y.backward()
print(x.grad)

# check whether all function(s) and variable(s) are remembering its output & creator
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x