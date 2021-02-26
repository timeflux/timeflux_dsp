import pandas as pd

from timeflux.core.node import Node


class Concat(Node):
    """Concat list of data ports .
    Attributes:
        i_* (Port): Dynamic inputs, expects DataFrame and meta.
        o (Port): Default output, provides DataFrame.
    Args:
        axis (str|int): The axis to concatenate along.
        kwargs: Keyword arguments to pass to pd.concat function

    Notes
    -----
    There is no shape verification in this node, that won't result in any Exception
    but may introduce NaN in the DataFrame. The responsability is left to the user.

    """

    def __init__(self, axis=1, **kwargs):
        self._axis = axis
        self._kwargs = kwargs

    def update(self):
        ports = list(self.iterate("i*"))
        i_data = [port.data for (name, _, port) in ports if port.data is not None]
        if i_data:
            self.o.data = pd.concat(i_data, axis=self._axis, **self._kwargs)
