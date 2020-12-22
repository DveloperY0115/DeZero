import numpy as np
import weakref
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


class Config:
    enable_backprop = True


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            # function only remembers input & output when it needs to do backward propagation
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx



""" Function wrappers """
def add(x0, x1):
    f = Add()
    return f(x0, x1)

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)
