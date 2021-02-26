import numpy as np
from timeflux.core.node import Node


class Discretize(Node):
    """Discretize data based on defined ranges
    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        range (dict): dictionary with keys corresponding to the discrete labels and values
                      are lists of tuple with boundaries.
        default (float|str|None):
    """

    def __init__(self, range, default=None):
        super().__init__()

        if default is None:
            default = np.NaN
        default = default
        self._default = default
        self._range = range
        for k, range_values in self._range.items():
            for v in range_values:
                if v[0] is None:
                    v[0] = -np.inf
                if v[1] is None:
                    v[1] = np.inf

    def _discretize(self, x):
        for k, range_values in self._range.items():
            for v in range_values:
                if (x >= v[0]) & (x <= v[1]):
                    return k
        return self._default

    def update(self):
        if self.i.ready():
            self.o = self.i
            if self.o.data is not None:
                self.o.data = self.o.data.applymap(lambda x: self._discretize(x))
