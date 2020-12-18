import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from core import Variable

data = np.array(1.0)
x = Variable.Variable(data)
print(x.data)