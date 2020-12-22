import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from core.Variable import Variable
from core.Function import square


def stupid_squares():
    for i in range(10):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))


if __name__ == '__main__':
    stupid_squares()
