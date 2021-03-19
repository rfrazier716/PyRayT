import numpy as np


class RGBAColor(np.ndarray):
    def __new__(cls, *args, **kwargs):
        # creates an array with the homogeneous coordinates
        obj = np.zeros(4, dtype=float).view(cls)
        return obj

    def __init__(self, r: float = 0, g: float = 0, b: float = 0, a: float = 1) -> None:
        # assign initialization
        self[0] = r
        self[1] = g
        self[2] = b
        self[3] = a

    @property
    def r(self):
        return self[0]

    @r.setter
    def r(self, x):
        self[0] = x

    @property
    def g(self):
        return self[1]

    @g.setter
    def g(self, y):
        self[1] = y

    @property
    def b(self):
        return self[2]

    @b.setter
    def b(self, z):
        self[2] = z

    @property
    def a(self):
        return self[3]

    @a.setter
    def a(self, w):
        self[3] = w


WHITE = RGBAColor(1, 1, 1)
BLACK = RGBAColor()
RED = RGBAColor(1, 0, 0)
GREEN = RGBAColor(0, 1, 0)
BLUE = RGBAColor(0, 0, 1)
YELLOW = RGBAColor(1, 1, 0)
ORANGE = RGBAColor(1, 0.5, 0)
