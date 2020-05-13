"""This module contains nodes for time index processing"""

import numpy as np

from timeflux.core.node import Node


class DelayIndex(Node):
    def __init__(self, delay=None):
        super().__init__()
        self._delay = None
        self._delay = self.to_timedelta(delay)

    def to_timedelta(self, duration):
        if duration is None:
            return None
        return np.timedelta64(1, "us") * (duration * 1e6)

    def update(self):
        if not self.i.ready():
            return
        if self._delay is None:
            self._delay = self.to_timedelta(self.o.meta.get("delay"))
        if self._delay is not None:
            self.o = self.i
            self.o.data.index -= self._delay
