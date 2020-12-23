import numpy as np
import heapq

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('Object of type {} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

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
