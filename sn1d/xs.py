from numba.experimental import jitclass
from numba import float64

spec = [
    ('total', float64[:]),  
    ('scatter', float64[:]),
    ('fission', float64[:]), 
    ('multiplicity', float64[:])
]

# class to hold all xs data

@jitclass(spec)
class CrossSection:

    # Constructor method to initialize instance attributes
    def __init__(self, total, scatter, fission, multiplicity):
        self.total = total 
        self.scatter = scatter
        self.fission = fission
        self.multiplicity = multiplicity

        if (len(total) != len(scatter)):
            raise ValueError(f"'total' and 'scatter' must be same length (got {len(total)} and {len(scatter)})")

        if (len(total) != len(fission)):
            raise ValueError(f"'total' and 'fission' must be same length (got {len(total)} and {len(fission)})")

        if (len(total) != len(multiplicity)):
            raise ValueError(f"'total' and 'multiplicity' must be same length (got {len(total)} and {len(multiplicity)})")