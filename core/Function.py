import numpy as np
from .Variable import Variable


def as_array(x):
    """
    Checks the type of input and convert it to numpy array
    :param x: Any
    :return: Instance of numpy array created using 'x'
    """
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx



""" Function wrappers """
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)
