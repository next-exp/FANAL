from __future__  import annotations

import numpy    as np
from   math import pi


class XYZ:

    def __init__(self,
                 x   : float,
                 y   : float,
                 z   : float) :
        self.x = x
        self.y = y
        self.z = z

    @classmethod
    def from_array(cls, array : np.array):
        "Initialize XYZ from a np.array"
        if len(array) != 3: raise IndexError
        return cls(array[0], array[1], array[2])

    def __str__(self):
        return f"(x = {self.x}, y = {self.y}, z = {self.z})"

    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.x
        if n == 1: return self.y
        if n == 2: return self.z
        raise IndexError

    @property
    def array(self): return np.array([self.x, self.y, self.z])

    @property
    def rad(self): return np.sqrt(self.x**2 + self.y**2)

    @property
    def phi(self): return np.arctan2(self.y, self.x)

    @property
    def phi_deg(self): return self.phi * 180. / pi

    @property
    def theta(self): return np.arctan2(self.rad, self.z)

    @property
    def theta_deg(self): return self.theta * 180. / pi

    def distance(self, point : XYZ):
        return np.sqrt((self.x - point.x)**2 +
                       (self.y - point.y)**2 +
                       (self.z - point.z)**2)


