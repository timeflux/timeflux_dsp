from timeflux.core.node import Node
import numpy as np
import pandas as pd


class Discretize(Node):
    """ Discretize data based on defined ranges
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


class Classifier(Node):
    """Node that converts data into arbitrary values in accordance with specified rules.

    The rules are provided as expressions which must evaluate to true or false. If data satisfies multiple rules, the
    last rules wins.

    See timeflux/nodes/expression.py for additional documentation on using expressions.

    Args:
        rules (dict): The classifications to impose on the data. The dictionary values are expressions. When an
            expression evaluates to True, the dictonary's key is set as the value in the 'classification' output.
        **kwargs (dict): Additional arguments to provide to pd.eval().

    Attributes:
        i (Port): (optional) Default input, may provide a pandas.DataFrame with columns which are passed on but not used in classification.
        i_* (Port): Inputs used in evaluating the classification rules.
        o (Port): Default output, provides a pandas.DataFrame with column 'classification'.

    Examples:

        .. code-block:: yaml

           graphs:
              - nodes:
                - id: weather
                  module: timeflux_dsp.nodes.squashing
                  class: Classifier
                  params:
                    rules:
                      cold: i_temperature < 5
                      wet: i_rainfall > 10
                      nasty: i_temperature < 5 and i_rainfall > 10
    """

    def __init__(self, rules, **kwargs):
        super().__init__()
        self._rules = rules
        self._kwargs = kwargs

    def update(self):
        self.o = self.i
        if self.o.data is None:
            self.o.data = pd.DataFrame(columns=['classification'])
        else:
            self.o.data['classification'] = pd.Series()

        for classification, rule in self._rules.items():
            # Perform the eval, and skip event if inputs are missing.
            calculated = self._eval_expression(rule)
            if calculated is None:
                continue

            # Trigger event only when all columns have satisfied expression.
            triggered = calculated.all(axis='columns')
            triggered = triggered[triggered == True]

            if triggered.shape[0] > 0:
                new_rows = list(triggered.index)
                idx = list(self.o.data.index)
                idx.extend(new_rows)
                self.o.data = self.o.data.reindex(index=idx)
                self.o.data.loc[new_rows,'classification'] = classification

    def _eval_expression(self, expression):
        ports = [port_name for port_name, _, _
                                in self.iterate('i_*')
                                if port_name in expression]
        _local_dict = {port_name: self.ports.get(port_name).data
                       for port_name in ports}
        if np.any([data is None for data in _local_dict.values()]):
            return
        calculated = pd.eval(expr=expression,
                              local_dict=_local_dict,
                              **self._kwargs)
        for port_name in ports:
            self.o.meta.update(self.ports.get(port_name).meta)
        return calculated