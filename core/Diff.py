from .Variable import Variable
from .Function import Function


def numerical_diff(f, x, eps=1e-4):
    """
    Calculate the differential coefficient of function 'f' at point 'x'
    :param f: Instance of function classes
    :param x: Instance of numpy array
    :param eps: Rate of change of 'x', default is 1e-4
    :return: df/dx at given point 'x'
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)
