import numpy as np
import weakref
import contextlib
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


# Context manager for no_grad mode
@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# Wrapper function for 'using_config('enable_backprop', False)'
def no_grad():
    return using_config('enable_backprop', False)


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

    def __lt__(self, other):
        return self.generation < other.generation

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


def add(x0, x1):
    f = Add()
    return f(x0, x1)


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)
