import numpy as np

from numba.experimental import jitclass
from numba import float64, int64

spec = [
    ('w', float64[:]),
    ('mu', float64[:]),
    ('M', int64)
]

# class to hold all xs data

@jitclass(spec)
class Quadrature:


    # Constructor method to initialize instance attributes
    def __init__(self, mu, w):
        epsilon = 1e-6
        self.mu = mu 
        self.w = w
        # ensure mu and w are the same length
        if len(mu) != len(w):
            raise ValueError(f"'mu and 'w' must be the same length (got {len(mu)} and {len(w)})")

        # ensure mu is ascending
        is_ascending = np.all(mu[:-1] <= mu[1:])
        if is_ascending == False:
            raise ValueError("'mu' must be ascending")

        # # ensure sum(w) = 2 within tolerance epsilon
        # sum_w = np.sum(w)
        # if np.abs(sum_w - 2) > epsilon:
        #     raise ValueError(f"sum(w) must be == 2 (got {sum_w:.8e})")

        self.M = len(mu)

    