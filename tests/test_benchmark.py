import unittest
import numpy as np
from DeZero import Variable


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein_price(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
        (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


class SphereTest(unittest.TestCase):
    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        # print how close the result is
        print('Testing Sphere...')
        print('Difference between calculated value and true gradient: {}'.format(abs(x.grad - 2.0)))
        print('Difference between calculated value and true gradient: {}\n'.format(abs(y.grad - 2.0)))
        self.assertEqual(x.grad, 2.0)
        self.assertEqual(y.grad, 2.0)


class MatyasTest(unittest.TestCase):
    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        # print how close the result is
        print('Testing Matyas...')
        print('Difference between calculated value and true gradient: {}'.format(abs(x.grad - 0.04)))
        print('Difference between calculated value and true gradient: {}\n'.format(abs(y.grad - 0.04)))
        flg1 = np.allclose(x.grad, 0.04)
        flg2 = np.allclose(y.grad, 0.04)
        self.assertTrue(flg1)
        self.assertTrue(flg2)


class Goldstein_PriceTest(unittest.TestCase):
    def test_goldstein_price(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein_price(x, y)
        z.backward()
        print('Testing Goldstein-Price...')
        print('dz/dx : ' + str(x.grad))
        print('dz/dy : ' + str(y.grad) + '\n')
