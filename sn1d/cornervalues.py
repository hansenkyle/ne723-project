import numpy as np
from numba.experimental import jitclass
from numba import float64

spec = [
    ('L', float64[:]), 
    ('R', float64[:])
]

@jitclass(spec)
class CornerValues:
    def __init__(self, L : np.ndarray, R: np.ndarray):
        self.L = L
        self.R = R

        if len(L) != len(R):
            raise ValueError(f"left and right CV vectors must have equal length (got {len(L)} and {len(R)})")

    def __sub__(self, other):
        if isinstance(other, CornerValues):
            return CornerValues(self.L - other.L, self.R - other.R)

    def __truediv__(self, scalar):
        if isinstance(scalar, float):
            return CornerValues(self.L/scalar, self.R/scalar)

    def copy(self):
        return CornerValues(self.L, self.R)
    
    def norm_inf(self):
        return np.max(np.abs(np.append(self.L, self.R)))
    
    def cell_average(self):
        return 0.5*(self.L + self.R)
    def first_moment(self):
        return 0.5*(self.R - self.L)