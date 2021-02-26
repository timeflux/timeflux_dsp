"""This module contains nodes for signal filtering."""

import numpy as np
import pandas as pd
from scipy import signal

from timeflux.core.branch import Branch
from timeflux.core.node import Node
from timeflux.nodes.window import TimeWindow
from timeflux_dsp.utils.filters import (
    construct_fir_filter,
    construct_iir_filter,
    design_edges,
)
from timeflux_dsp.utils.import_helpers import make_object


class DropRows(Node):
    """Decimate signal by an integer factor.

    This node uses Pandas computationally efficient functions to drop rows.
    By default, it simply transfers one row out of ``factor`` and drops the others.
    If ``method`` is `mean` (resp. median), it applies a rolling window of length
    equals ``factor``, computes the mean and returns one value per window.
    It maintains an internal state to ensure that every k'th sample is picked
    even across chunk boundaries.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        factor (int): Decimation factor. Only every k'th sample will be
        transferred into the output.
        method (str|None): Method to use to drop rows.
                           If `None`, the values are transferred as it.
                           If `mean` (resp. `median`), the mean (resp. median)
                           of the samples is taken.

    Example:
       .. literalinclude:: /../examples/droprows.yaml
           :language: yaml

    Example:
        In this exemple, we generate white noise to stream and we drop one sample out
        of two using DropRows, setting:

        * ``factor`` = `2`
        * ``method`` = `None` (see orange trace) | ``method`` = `"mean"` (see green trace)

        .. image:: /static/image/droprows_io.svg
           :align: center


    Notes:
        Note that this node is not supposed to dejitter the timestamps, so if
        the input chunk is not uniformly sampled, the output chunk won’t be either.

        Also, this filter does not implement any anti-aliasing filter.
        Hence, it is recommended to precede this node by a low-pass filter
        (e.g., FIR or IIR) which cuts out below half of the new sampling rate.

    """

    def __init__(self, factor, method=None):

        super().__init__()
        self._factor = factor
        self._method = method
        self._previous = pd.DataFrame()

    def update(self):

        # copy the meta
        self.o.meta = self.i.meta

        # if nominal rate is specified in the meta, update it.
        if "rate" in self.o.meta:
            self.o.meta["rate"] /= self._factor

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        self.i.data = pd.concat([self._previous, self.i.data], axis=0, sort=True)

        n = self.i.data.shape[0]
        remaining = n % self._factor
        self.i.data, self._previous = np.split(self.i.data, [n - remaining])

        if self._method is None:
            # take every kth sample with k=factor starting from the k-1 position
            self.o.data = self.i.data.iloc[self._factor - 1 :: self._factor]
        else:
            # estimate rolling mean (or median) with window length=factor and take
            # every kth sample with k=factor starting from the k-1 position
            if self._method == "mean":
                self.o.data = (
                    self.i.data.rolling(
                        window=self._factor, min_periods=self._factor, center=False
                    )
                    .mean()
                    .iloc[self._factor - 1 :: self._factor]
                )
            elif self._method == "median":
                self.o.data = (
                    self.i.data.rolling(
                        window=self._factor, min_periods=self._factor, center=False
                    )
                    .median()
                    .iloc[self._factor - 1 :: self._factor]
                )


class Resample(Node):
    """Resample signal.

    This node calls the `scipy.signal.resample` function to decimate the signal
    using Fourier method.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        factor (int): Decimation factor. Only every k'th sample will be
                      transferred into the output.
        window (str|list|float): Specifies the window applied to the signal
                      in the Fourier domain. Default: `None`.

    Example:
        .. literalinclude:: /../examples/resample.yaml
           :language: yaml

    Notes:
        This node should be used after a buffer to assure that the FFT window
        has always the same length.

    References:

        * `scipy.signal.resample <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.resample.html>`_

    """

    def __init__(self, factor, window=None):

        super().__init__()
        self._factor = factor
        self._window = window
        self._previous = pd.DataFrame()

    def update(self):

        # copy the meta
        self.o.meta = self.i.meta

        # if nominal rate is specified in the meta, update it.
        if "rate" in self.o.meta:
            self.o.meta["rate"] /= self._factor

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        n = self.i.data.shape[0]

        if not self._previous.empty:
            self.i.data = pd.concat([self._previous, self.i.data], axis=0)

        if self.i.data.shape[0] % self._factor == 0:
            self._previous = pd.DataFrame()
        else:
            self._previous = self.i.data.iloc[(n // self._factor) * self._factor :]
            self.i.data = self.i.data.iloc[: (n // self._factor) * self._factor]

        self.o.data = pd.DataFrame(
            data=signal.resample(
                x=self.i.data.values, num=n // self._factor, window=self._window
            ),
            index=self.i.data.index[np.arange(0, n - 1, self._factor)],
            columns=self.i.data.columns,
        )


class IIRFilter(Node):
    """Apply IIR filter to signal.

    If ``sos`` is `None`, this node uses adapted methods from mne.filters to
    design the filter coefficients based on the specified parameters.
    If no transition band is given, default is to use :

    * l_trans_bandwidth =  min(max(l_freq * 0.25, 2), l_freq)
    * h_trans_bandwidth =   min(max(h_freq * 0.25, 2.), rate / 2. - h_freq)

    Else, it uses ``sos`` as filter coefficients.

    Once the kernel has been estimated, the node applies the filtering to each
    columns in ``columns`` using `scipy.signal.sosfilt` to generate the output given the input,
    hence ensures continuity  across chunk boundaries,

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
                        from the meta.
        order (int, optional): Filter order. Default: `None`.
        frequencies (list|None): Transition frequencies. Ignored when sos is given.
        filter_type (str|None): Filter mode (`lowpass`, `highpass`, `bandstop`, `bandpass`).
                        Default: `bandpass`. Ignored when sos is given.
        sos (array|None, optional) : Array of second-order sections (sos) representation,
                                    must have shape (n_sections, 6). Default: `None`.
        kwargs: keyword arguments to pass to the filter constructor


    Example:
        In this example, we generate a signal that is the sum of two sinus with
        respective periods of 1kHz and 15kHz and respective amplitudes of 1 and 0.5.
        We stream this signal using the IIRFilter node, designed for lowpass
        filtering at cutoff frequency 6kHz, order 3.

        * ``order`` = `3`
        * ``freqs`` = `[6000]`
        * ``mode`` = `'lowpass'`

        We plot the input signal, the output signal and the corresponding offline filtering.

        .. image:: /static/image/iirfilter_io.svg
           :align: center

    Notes:
        This node ensures continuity across chunk boundaries, using a recursive algorithm,
        based on a cascade of biquads filters.

        The filter is initialized to have a minimal step response, but needs a
        'warm up' period for the filtering to be stable, leading to small artifacts
        on the first few chunks.

        The IIR filter is faster than the FIR filter and delays the signal less
        but this delay is not constant and the stability not  ensured.

    References:

            * `Real-Time IIR Digital Filters <http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ece5655_chap8.pdf>`_
            * `scipy.signal.sosfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html>`_

    """

    def __init__(
        self, frequencies=None, rate=None, filter_type="bandpass", sos=None, **kwargs
    ):

        super().__init__()
        # self._order = order
        self._frequencies = frequencies
        self._filter_type = filter_type
        self._kwargs = dict(order=None, design="butter", pass_loss=3.0, stop_atten=50.0)
        self._kwargs.update(kwargs)
        self._rate = rate
        self._zi = None
        self._sos = None
        self._sos_custom = sos
        self._columns = None

    def update(self):

        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        if self._columns is None:
            self._columns = self.i.data.columns

        # set rate from the data if it is not yet given
        if self._rate is None:
            self._rate = self.i.meta.get("rate", None)
            if self._rate is None:
                # If there is no rate in the meta, set rate to 1.0
                self._rate = 1.0
                self.logger.warning(
                    f"Nominal rate not supplied, considering " f"1.0 Hz instead. "
                )
            else:
                self.logger.info(f"Nominal rate set to {self._rate}. ")

        if self._sos is None:
            self._design_sos()
        if self._zi is None:
            zi0 = signal.sosfilt_zi(self._sos)
            self._zi = np.stack(
                [
                    (zi0 * self.i.data.iloc[0, k_col])
                    for k_col in range(len(self._columns))
                ],
                axis=1,
            )
        port_o, self._zi = signal.sosfilt(self._sos, self.i.data.values.T, zi=self._zi)
        self.o.data = pd.DataFrame(
            port_o.T, columns=self._columns, index=self.i.data.index
        )

    def _design_sos(self):

        if self._sos_custom is None:
            # Calculate an IIR filter kernel for a given sampling rate.
            self._sos, self._freqs = construct_iir_filter(
                rate=self._rate,
                frequencies=self._frequencies,
                filter_type=self._filter_type,
                output="sos",
                **self._kwargs,
            )
        else:
            if self._sos_custom.shape[1] == 6:
                self._sos = self._sos_custom
            else:
                raise ValueError(
                    f"sos must have shape (n_sections, 6), received {self._sos_custom.shape} instead. "
                )


class IIRLineFilter(Node):
    """Apply multiple Notch IIR Filter in series.

        Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
                  from the meta.
        edges_center: List with center of the filters.
        orders (tuple|int|None): List with orders of the filters.
                            If int, the same order will be used for all filters.
                            If None, order 2 will be used for all filters.
        edges_width: List with orders of the filters.
                     If int, the same order will be used for all filters.
                     If None, width of 3 (Hz) will be used for all filters.

    """

    def __init__(
        self,
        rate=None,
        edges_center=(50, 60, 100, 120),
        orders=(2, 1, 1, 1),
        edges_width=(3, 3, 3, 3),
    ):

        super().__init__()

        orders = orders or 2
        edges_width = edges_width or 3

        if isinstance(orders, int):
            orders = [orders] * len(edges_center)

        if isinstance(edges_width, int):
            edges_width = [edges_width] * len(edges_center)

        filter_type = "bandstop"
        self._nodes = []
        for edge_center, edge_width, order in zip(edges_center, edges_width, orders):
            frequencies = [edge_center - edge_width, edge_center + edge_width]
            self._nodes.append(
                IIRFilter(
                    rate=rate,
                    order=order,
                    frequencies=frequencies,
                    filter_type=filter_type,
                )
            )

    def update(self):

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return
        # At this point, we are sure that we have some data to process
        # initialize output port
        self.o = self.i
        # apply each filter in series
        for node in self._nodes:
            node.i.data = self.o.data
            node.update()
            self.o.data = node.o.data


class FIRFilter(Node):
    """Apply FIR filter to signal.

    If ``coeffs`` is `None`, this node uses adapted methods from *mne.filters*
    to design the filter coefficients based on the specified parameters.
    If no transition band is given, default is to use:

    * l_trans_bandwidth =  min(max(l_freq * 0.25, 2), l_freq)
    * h_trans_bandwidth =   min(max(h_freq * 0.25, 2.), fs / 2. - h_freq)

    Else, it uses ``coeffs`` as filter coefficients.

    It applies the filtering to each columns in ``columns`` using `scipy.signal.lfilter`
    to generate the output given the input,
    hence ensures continuity  across chunk boundaries,

    The delay introduced is estimated and stored in the meta ``FIRFilter``, ``delay``.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame and meta.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
                      from the meta.
        columns (list|'all', optional): Columns to apply filter on. Default: `all`.
        order (int): Filter order.
        frequencies (list): Transition frequencies.
        filter_type (str, optional): Filter mode (`lowpass`, `highpass`, `bandstop`
                                    or `bandpass`). Default: `bandpass`.
        coeffs (array|None, optional): Custom coeffs to pass as ``b`` in `signal.filter`.
                                        Default: `None`.
        kwargs: keyword arguments to pass to the filter constructor (window, phase,... )

    Example:
        In this exemple, we generate a signal that is the sum of two sinus with
        respective periods of 1kHz and 15kHz and respective amplitudes of 1 and 0.5.
        We stream this signal using the FIRFilter node, designed for lowpass
        filtering at cutoff frequency 6kHz, order 20.

        * ``order`` = `20`
        * ``freqs`` = `[6000, 6100]`
        * ``mode`` = `'lowpass'`

        The FIR is a linear phase filter, so it allows one to correct for the
        introduced delay. Here, we retrieve the input sinus of period 1kHz.
        We plot the input signal, the output signal, the corresponding offline
        filtering and the output signal after delay correction.

        .. image:: /static/image/firfilter_io.png
           :align: center

    Notes:
        The FIR filter ensures a linear phase response, but is computationnaly
        more costly than the IIR filter.

        The filter is initialized to have a minimal step response, but needs a
        'warmup' period for the filtering to be stable, leeding to small artifacts on
        the first few chunks.

    """

    def __init__(
        self,
        frequencies,
        rate=None,
        columns="all",
        order=20,
        filter_type="bandpass",
        coeffs=None,
        **kwargs,
    ):

        super().__init__()
        self._order = order
        self._frequencies = frequencies
        self._mode = filter_type
        self._rate = rate
        self._columns = columns if columns != "all" else None
        self._coeffs_custom = coeffs
        self._kwargs = dict(design="firwin2", phase="linear", window="hamming")
        self._kwargs.update(kwargs)

        # Initialize the filter kernels and states, one per stream
        self._zi = {}  # FIR filter states, one per column
        self._coeffs = None  # FIR filter coeffs
        self._delay = None  # FIR filter delays

    def update(self):

        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        if self._columns is None:
            self._columns = self.i.data.columns

        # set rate from the data if it is not yet given
        if self._rate is None:
            self._rate = self.i.meta.get("rate", None)
            if self._rate is None:
                # If there is no rate in the meta, set rate to 1.0
                self._rate = 1.0
                self.logger.warning(
                    f"Nominal rate not supplied, considering " f"1.0 Hz instead. "
                )
            else:
                self.logger.info(f"Nominal rate set to {self._rate}. ")

        self._coeffs, self._delay = self._design_filter()

        for column in self._columns:
            if column not in self._zi:
                zi0 = signal.lfilter_zi(self._coeffs, 1.0)
                self._zi[column] = zi0 * self.i.data[column].values[0]
            port_o_col, self._zi[column] = signal.lfilter(
                b=self._coeffs,
                a=1.0,
                x=self.i.data[column].values.T,
                zi=self._zi[column],
            )
            # self.o.meta.update({'FIRFilter': {'delay': self._delay}})
            self.o.data.loc[:, column] = port_o_col
            # update delay
            delay = self.o.meta.get("delay") or 0.0
            delay += self._delay
            self.o.meta.update({"delay": delay})

    def _design_filter(self):
        """Calculate an FIR filter kernel for a given sampling rate."""

        nyq = self._rate / 2.0

        if self._coeffs_custom is None:
            edges, gains, _, _ = design_edges(
                frequencies=self._frequencies, nyq=nyq, mode=self._mode
            )
            fir_coeffs = construct_fir_filter(
                self._rate, edges, gains, self._order, **self._kwargs
            )
        else:
            fir_coeffs = self._coeffs_custom
        warmup = self._order - 1
        fir_delay = (warmup / 2) / self._rate
        return fir_coeffs, fir_delay


class Scaler(Node):
    """Apply a sklearn scaler

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame and meta.

    Args:
        method (str): Name of the scaler object (see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
        **kwargs: keyword arguments  to initialize the scaler.

    """

    def __init__(self, method="sklearn.preprocessing.StandardScaler", **kwargs):

        super().__init__()
        try:
            self._scaler = make_object(method, kwargs)
        except AttributeError:
            raise ValueError("Cannot make object from {method}".format(method=method))

    def update(self):

        if not self.i.ready():
            return

        # scale the signal
        self.o.data = pd.DataFrame(
            data=self._scaler.fit_transform(self.i.data.values),
            columns=self.i.data.columns,
        )
        if len(self.o.data) == len(self.i.data):
            self.o.data.index = self.i.data.index


class AdaptiveScaler(TimeWindow):
    """Scale the data adaptively.
    This nodes transforms the data using a sklearn scaler object that is continuously fitted on a rolling window.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame and meta.

    Args:
       length (float): The length of the window, in seconds.
       method (str): Name of the scaler object (see https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)
       dropna (bool): Whether or not NaN should be dropped before fitting the estimator. Default to False.
       **kwargs : keyword arguments  to initialize the scaler.
    """

    def __init__(
        self,
        length,
        method="sklearn.preprocessing.StandardScaler",
        dropna=False,
        **kwargs,
    ):

        super().__init__(length=length, step=0)
        self._fitted = False
        self._dropna = dropna
        try:
            self._scaler = make_object(method, kwargs)
        except AttributeError:
            raise ValueError(f"Module sklearn.preprocessing has no object {method}")

    def update(self):

        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        super().update()

        # if the window output is ready, fit the scaler with its values
        if self.o.ready():
            x = self.o.data.values
            if self._dropna:
                x = x[~np.isnan(x)]
            self._scaler.fit(x)
            self._fitted = True
            self.o.clear()

        # copy meta
        self.o.meta = self.i.meta
        # if the scaler has been fitted, transform the current data
        if self._fitted and self.i.ready():
            transformed_data = self._scaler.transform(self.i.data.values)
            self.o.data = pd.DataFrame(
                data=transformed_data,
                columns=self.i.data.columns,
                index=self.i.data.index,
            )


class FilterBank(Branch):
    """Apply multiple IIR Filters to the signal and stack the components horizontally

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
            from the meta.
        filters (dict|None): Define the iir filter to apply given its name and its params.
    """

    def __init__(self, filters, method="IIRFilter", rate=None, **kwargs):
        super().__init__()
        self._filters = filters

        graph = {"nodes": [], "edges": []}
        graph["nodes"].append(
            {
                "id": "stack",
                "module": "timeflux_dsp.nodes.helpers",
                "class": "Concat",
            }
        )

        for filter_name, filter_params in self._filters.items():
            filter_params.update({"rate": rate})
            filter_params.update(kwargs)
            iir = {
                "id": filter_name,
                "module": "timeflux_dsp.nodes.filters",
                "class": method,
                "params": filter_params,
            }
            rename_columns = {
                "id": f"rename_{filter_name}",
                "module": "timeflux.nodes.axis",
                "class": "AddSuffix",
                "params": {"suffix": f"_{filter_name}"},
            }
            graph["nodes"] += [iir, rename_columns]
            graph["edges"] += [
                {"source": filter_name, "target": f"rename_{filter_name}"},
                {"source": f"rename_{filter_name}", "target": f"stack:{filter_name}"},
            ]

        self.load(graph)

    def update(self):
        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return
        # set the data in input of each filter
        for filter_name in self._filters.keys():
            self.set_port(filter_name, port_id="i", data=self.i.data, meta=self.i.meta)

        self.run()

        self.o = self.get_port("stack", port_id="o")
