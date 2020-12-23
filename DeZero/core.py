import numpy as np
import heapq
import weakref
import contextlib


"""
Implementation of Variable class
"""


class Variable:

    __array_priority__ = 200    # to prevent calling magic method defined by numpy

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('Object of type {} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __len__(self):
        return len(self.data)

    def __neg__(self):
        return neg(self)

    def __pow__(self, power, modulo=None):
        return pow(self, power)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                # funcs.append(f)
                heapq.heappush(funcs, (-f.generation, f))
                seen_set.add(f)
                # TODO: Implement this using priority queue rather than sorting
                # funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = heapq.heappop(funcs)[1]
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                # if user decides not to retain gradients of variables in the middle, remove them
                for y in f.outputs:
                    y().grad = None


"""
Implementation of Function class
"""


def as_array(x):
    """
    Checks the type of input and convert it to numpy array
    :param x: Any
    :return: Instance of numpy array created using 'x'
    """
    if np.isscalar(x):
        return np.array(x)
    return x


def as_variable(x):
    """
    Checks the type of input and convert it to numpy array
    :param x: Any
    :return: Instance of Variable created using 'x'
    """
    if not isinstance(x, Variable):
        if not isinstance(x, np.ndarray):
            return Variable(as_array(x))
        return Variable(x)
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
        inputs = [as_variable(x) for x in inputs]

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


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def neg(x):
    f = Neg()
    return f(x)


def square(x):
    f = Square()
    return f(x)


def pow(x, c):
    f = Pow(c)
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)


def sub(x0, x1):
    f = Sub()
    return f(x0, x1)


def mul(x0, x1):
    f = Mul()
    return f(x0, x1)


def div(x0, x1):
    f = Div()
    return f(x0, x1)
