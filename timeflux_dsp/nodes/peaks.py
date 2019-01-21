import numpy as np
import pandas as pd
from timeflux.core.io import Port
from timeflux.core.node import Node
from timeflux.helpers.clock import *
from datetime import timedelta


class RealTimeDetect(Node):
    """ Detect peaks and valleys in live 1D signal
    This node uses a simple algorithm to detect peaks in real time.
    When a local extrema (peak or valley) is detected, an event is sent with the nature specified in the label column ("peak"/"valley) and
    the characteristics in the data column, giving:

    - **value**: Amplitude of the extrema.
    - **lag**: Time laps between the extrema and its detection.
    - **interval**: Duration between two extrema os same nature.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o_events (Port): Events output, provides DataFrame.

    Example:
       .. literalinclude:: /../../timeflux_dsp/test/graphs/droprows.yaml
           :language: yaml

    Example:
        In this example, we stream a photoplethysmogram signal scaled between -1 and 1 and we use the node RealTimeDetect to detect peaks and valleys.

        * ``delta`` = `0.1`
        * ``tol`` = `0.5`
        * ``reset`` = None

        .. image:: /../../timeflux_dsp/doc/static/image/realtimepeaks_io.svg
           :align: center

        self.o_events.data::

                                            label                                               data
            2018-11-19 11:06:39.620900000    peak  {'value': [1.0054607391357422], 'lag': 0.03125, 'interval': 0.654236268}
            2018-11-19 11:06:39.794709043  valley  {'value': [-1.0110111236572266], 'lag': 0.046875, 'interval': 0.654236268}
            2018-11-19 11:06:40.605209027    peak  {'value': [0.9566234350204468],  'lag': 0.033197539, 'interval': 0.984309027}
            2018-11-19 11:06:40.761455675  valley  {'value': [-1.0549353361129759], 'lag': 0.048816963, 'interval': 0.810499984}




    Notes:

        This peak detection is considered real-time since it does not require buffering the data. However, the detection method necessarily involve a lag between the actual peak and its detection.
        Indeed, the internal state of the node is either "looking for a peak" or "looking for a valley".

        - If the node is "looking for a peak", it means it computes local maxima until the signal drops significantly (ie. more than ``delta``)
        - If the node is "looking for a valley", it means it computes local minima until the signal rises significantly (ie. more than ``delta``)

        The "last local extrema" is set to a peak (resp. valley) as soon as the signal drops (resp. rises) significantly.
        Hence, there is an intrinsic lag in the detection, that is directly linked to parameter ``delta``.

        Hence, by decreasing delta, we minimize the lag. But if delta is too small, we'll suffer from false positive detection, unless ``tol`` is tuned to avoid too closed detections.
        The parameters should be tuned depending on the nature of the data, ie. their dynamic, quality, shapes.

        See the illustration above:

        .. image:: /../../timeflux_dsp/doc/static/image/realtimepeaks_illustration.png
            :align: center


    References:

        * `Matlab function <http://billauer.co.il/peakdet.html>`_
        * `Publication <https://www.ncbi.nlm.nih.gov/pubmed/27027672>`_

    **To Do**:

        * allow for adaptive parametrization.
    """
    def __init__(self, delta, tol, reset=None):
        """
        Args:
            delta (float): Threshold for peak/valley matching in amplitude. This can be seen as the minimum significant change enough to detect a peak/valley.
            tol (float): Tolerence for peak/valley matching, in seconds. This can be seen as the minimum time difference between two peaks/valleys.
            reset (float): Reset threshold, in seconds. This can be seen as the maximum duration of plausible transitions between peaks and valleys. Default: None.
        """

        self._delta = delta  # Peak threshold
        self._tol = tol  # Tolerence for peak matching, in seconds. This can be seen as the minimum time difference between two peaks
        self._reset_states()
        self._last = pd.to_datetime(now())  # Last timestamp
        self._reset = reset
        self.o_events = Port()

    def _reset_states(self):
        """Reset peak detection internal state."""
        self._mxv = -np.Inf  # local max value
        self._mnv = np.Inf  # local min value
        self._mxt = None  # local max time
        self._mnt = None  # local min time
        self._lfm = True  # look for max

        self._last_peak = None
        self._last_valley = None

    def update(self):

        # copy the meta
        self.o.meta = self.i.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        self.o_events.data = None

        for (value, timestamp) in zip(self.i.data.values, self.i.data.index):
            if self._reset is not None:
                if (self._last - timestamp).total_seconds() > self._reset: self._reset_states()
            self._last = timestamp
            # Peak detection
            detected = self._on_sample(value=value, timestamp=timestamp)
            # Append event
            if detected:
                self.o_events.data = pd.DataFrame(index=[detected[0]],
                                           data=np.array([[detected[1]],
                                                          [{'value': detected[2], 'lag': detected[3],
                                                            'interval': detected[4]}]]).T,
                                           columns=['label', 'data'])

    def _on_sample(self, value, timestamp):
        """Peak detection"""
        if self._last_peak is None: self._last_peak = timestamp
        if self._last_valley is None: self._last_valley = timestamp

        if value > self._mxv:
            self._mxv = value
            self._mxt = timestamp

        if value < self._mnv:
            self._mnv = value
            self._mnt = timestamp
        if self._lfm:
            if (value < self._mxv - self._delta):

                _interval = (self._mxt - self._last_peak).total_seconds()
                # print(60/_interval)
                self._mnv = value
                self._mnt = timestamp
                self._lfm = False
                if ((self._mxt - self._last_peak).total_seconds() > self._tol):
                    self._last_peak = self._mxt
                    return (self._mxt, 'peak', self._mxv, (timestamp - self._mxt).total_seconds(), _interval)
        else:
            if (value > self._mnv + self._delta):
                _interval = (self._mxt - self._last_valley).total_seconds()
                self._mxv = value
                self._mxt = timestamp
                self._lfm = True
                if ((self._mnt - self._last_valley).total_seconds() > self._tol):
                    self._last_valley = self._mnt
                    return (self._mnt, 'valley', self._mnv, (timestamp - self._mnt).total_seconds(), _interval)
        return False


class WindowDetect(Node):
    """ Detect peaks and valleys on a rolling window of analysis in  1D signal
    This node uses a buffer to compute local extrema and detect peaks in real time.
    When a local extrema (peak or valley) is detected, an event is sent with the nature specified in the label column ("peak"/"valley) and
    the characteristics in the data column, giving:

    - **value**: Amplitude of the extrema.
    - **lag**: Time laps between the extrema and its detection.
    - **interval**: Duration between two extrema os same nature.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o_events (Port): Events output, provides DataFrame.
    """

    def __init__(self, window=0.5, tol=0.1):
        """
        Args:
            window (float): Window of analysis in seconds, on which local max/min is computed.
            tol (float): Tolerence for peak/valley matching, in seconds. This can be seen as the minimum time difference between two peaks/valleys.
        """

        self._window = window  # Window of analysis
        self._tol = tol  # Tolerence for peak matching, in seconds. This can be seen as the minimum time difference between two peaks

        self.o_events = Port()

        self._reset_states()

    def _reset_states(self):
        # Reset peak detection internal state.

        self._buffer = pd.DataFrame()
        # self._timestamps_buff = []  # np.zeros((2*self.n))
        self._last_peak = None
        self._last_valley = None

        self._peak_interval = timedelta(seconds=1)
        self._valley_interval = timedelta(seconds=1)

        self._warmed_up = False

    def update(self):

        # copy the meta
        self.o_events.meta = self.i.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        if not self._warmed_up:
            if self._last_peak is None: self._last_peak = self.i.data.index[0]
            if self._last_valley is None: self._last_valley = self.i.data.index[0]
            self._buffer = pd.concat([self._buffer, self.i.data], axis=0)
            if (self._buffer.index[-1] - self._buffer.index[0]).total_seconds() > 3 * self._window: self._warmed_up = True
        else:
            self.o_events.data = None
            ## append data to buffer
            self._buffer = pd.concat([self._buffer, self.i.data], axis=0)
            ## throw out old samples
            self._buffer = self._buffer.loc[self._buffer.index[-1] - timedelta(seconds=self._window):]
            # iterates over each sample
            # this is actually equivalent to the local max function of scipy
            if self.i.data.max()[0] == self._buffer.max()[0]:
                peak = self.i.data.idxmax()[0]
                if (peak - self._last_peak).total_seconds() > self._tol:
                    self._peak_interval = peak - self._last_peak

                    self.o_events.data = pd.DataFrame(index=[peak],
                                               data=np.array([['peak'],
                                                              [{'value': self.i.data.max()[0], 'lag': (
                                                                          self.i.data.index[
                                                                              -1] - self._last_peak).total_seconds(),
                                                                'interval': self._peak_interval.total_seconds()}]]).T,
                                               columns=['label', 'data'])
                    self._last_peak = peak

            if self.i.data.min()[0] == self._buffer.min()[0]:
                valley = self.i.data.idxmin()[0]
                if (valley - self._last_valley).total_seconds() > self._tol:
                    self._valley_interval = valley - self._last_valley

                    self.o_events.data = pd.DataFrame(index=[valley],
                                               data=np.array([['valley'],
                                                              [{'value': self.i.data.min()[0], 'lag': (
                                                                          self.i.data.index[
                                                                              -1] - self._last_valley).total_seconds(),
                                                                'interval': self._valley_interval.total_seconds()}]]).T,
                                               columns=['label', 'data'])
                    self._last_valley = valley


class Rate(Node):
    """ Computes rate of an event given its label.

    This node computes the inverse duration (ie. instantaneous rate) between onsets of successive events with a marker matching the ``event_trigger`` in the ``event_label`` column of the event input,

    Attributes:
        i_events (Port): Events input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    """
    def __init__(self, event_trigger='peak', event_label= 'label'):
        """
        Args:
            event_trigger (string): The marker name.
            event_label (string): The column to match for event_trigger.
        """

        self._event_trigger = event_trigger
        self._event_label = event_label
        self._reset_states()

    def _reset_states(self):
        # Reset peak detection internal state.

        self._last = None

    def update(self):

        # copy the meta
        self.o.meta = self.i_events.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process

        self.o.data = None

        target_index = self.i_events.data[self.i_events.data[
                                          self._event_label] == self._event_trigger].index

        if not target_index.empty:

            if self._last is None: self._last = target_index[0]

            target_index = [self._last] + list(target_index)

            rate_values = [(np.timedelta64(1, 's') / a) if (a != 0 * np.timedelta64(1, 's')) else None for a in
             list(np.diff(target_index))]

            self.o.data = pd.DataFrame(index = target_index[1:], data = rate_values, columns = ["rate"])
            self._last = target_index[-1]