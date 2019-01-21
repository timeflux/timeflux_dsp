"""This module contains nodes for signal filtering."""

from timeflux.core.node import Node
from scipy import signal
from timeflux.helpers.clock import *
from timeflux.core.io import Port

from ..utils.filters import construct_fir_filter, _get_com_factor, construct_iir_filter, design_edges


class DropRows(Node):
    """Decimate signal by an integer factor.

    This node uses Pandas computationally efficient functions to drop rows.
    By default, it simply transfers one row out of ``factor`` and drops the others.
    If ``method`` is `mean` (resp. median), it applies a rolling window of length equals ``factor``, computes the mean and returns one value per window.
    It maintains an internal state to ensure that every k'th sample is picked even across chunk boundaries.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
       .. literalinclude:: /../../timeflux_dsp/test/graphs/droprows.yaml
           :language: yaml

    Example:
        In this exemple, we generate white noise to stream and we drop one sample out of two using DropRows, setting:

        * ``factor`` = `2`
        * ``method`` = `None` (see orange trace) | ``method`` = `"mean"` (see green trace)

        .. image:: /../../timeflux_dsp/doc/static/image/droprows_io.svg
           :align: center


    Notes:
        Note that this node is not supposed to dejitter the timestamps, so if the input chunk is not uniformly sampled, the output chunk won’t be either.

        Also, this filter does not implement any anti-aliasing filter. Hence, it is recommended to precede this node by
        a low-pass filter (e.g., FIR or IIR) which cuts out below half of the new sampling rate.

    """

    def __init__(self, factor, method=None):
        """
         Args:
            factor (int): Decimation factor. Only every k'th sample will be transferred into the output.
            method (str|None): Method to use to drop rows. If `None`, the values are transferred as it. If `mean` (resp. median),
                               the mean (resp. median) of the samples is taken.
        """
        self._factor = factor
        self._method = method
        self._previous = pd.DataFrame()

    def update(self):

        # copy the meta
        self.o.meta = self.i.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        self.i.data = pd.concat([self._previous, self.i.data], axis=0)

        if self.i.data.shape[0] % self._factor == 0:
            self._previous = pd.DataFrame()
        else:
            self._previous = self.i.data.iloc[(self.i.data.shape[0] // self._factor) * self._factor:]
            self.i.data = self.i.data.iloc[: (self.i.data.shape[0] // self._factor) * self._factor]

        if self._method is None:
            # take every kth sample with k=factor starting from the k-1 position
            self.o.data = self.i.data.iloc[self._factor - 1::self._factor]
        else:
            # estimate rolling mean (or median) with window length=factor and take every kth sample with k=factor starting from the k-1 position
            if self._method == "mean":
                self.o.data = self.i.data.rolling(window=self._factor, min_periods=self._factor,
                                        center=False).mean().iloc[self._factor - 1::self._factor]
            elif self._method == "median":
                self.o.data = \
                    self.i.data.rolling(window=self._factor, min_periods=self._factor,
                                        center=False).median().iloc[self._factor - 1::self._factor]


class Resample(Node):
    """Resample signal.

    This node calls the `scipy.signal.resample` function to decimate the signal using Fourier method.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        .. literalinclude:: /../../timeflux_dsp/test/graphs/resample.yaml
           :language: yaml

    Notes:
        This node should be used after a buffer to assure that the FFT window has always the same length.

    References:

        * `scipy.signal.resample <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.signal.resample.html>`_

    """

    def __init__(self, factor, window=None):

        """
        Args:
            factor (int): Decimation factor. Only every k'th sample will be transferred into the output.
            window (str|list|float): Specifies the window applied to the signal in the Fourier domain. Default: `None`.
        """

        self._factor = factor
        self._window = window
        self._previous = pd.DataFrame()

    def update(self):

        # copy the meta
        self.o.meta = self.i.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        if not self._previous.empty:
            self.i.data = pd.concat([self._previous, self.i.data], axis=0)

        if self.i.data.shape[0] % self._factor == 0:
            self._previous = pd.DataFrame()
        else:
            self._previous = self.i.data.iloc[(self.i.data.shape[0] // self._factor) * self._factor:]
            self.i.data = self.i.data.iloc[: (self.i.data.shape[0] // self._factor) * self._factor]

        self.o.data = pd.DataFrame(
            data=signal.resample(x=self.i.data.values, num=self.i.data.shape[0] // self._factor,
                                 window=self.window),
            index=self.i.data.index[np.arange(0, self.i.data.shape[0], self._factor)],
            columns=self.i.data.columns)


class IIRFilter(Node):
    """Apply IIR filter to signal.

    If ``sos`` is `None`, this node uses adapted methods from mne.filters to design the filter coefficients based on the specified parameters.
    If no transition band is given, default is to use :

    * l_trans_bandwidth =  min(max(l_freq * 0.25, 2), l_freq)
    * h_trans_bandwidth =   min(max(h_freq * 0.25, 2.), fs / 2. - h_freq)

    Else, it uses ``sos`` as filter coefficients.

    Once the kernel has been estimated, the node applies the filtering to each columns in ``columns`` using `scipy.signal.sosfilt` to generate the output given the input,
    hence ensures continuity  across chunk boundaries,

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Example:
        In this example, we generate a signal that is the sum of two sinus with respective periods of 1kHz and 15kHz and respective amplitudes of 1 and 0.5.
        We stream this signal using the IIRFilter node, designed for lowpass filtering at cutoff frequency 6kHz, order 3.

        * ``order`` = `3`
        * ``freqs`` = `[6000]`
        * ``mode`` = `"lowpass"`

        We plot the input signal, the output signal and the corresponding offline filtering.

        .. image:: /../../timeflux_dsp/doc/static/image/iirfilter_io.svg
           :align: center

    Notes:
        This node ensures continuity across chunk boundaries, using a recursive algorithm, based on a cascade of biquads filters.

        The filter is initialized to have a minimal step response, but needs a "warmup" period for the filtering to be stable, leeding to small artifacts on the first few chunks.

        The IIR filter is faster than the FIR filter and delays the signal less but this delay is not constant and the stability not guarenteed.

    References:

            * `Real-Time IIR Digital Filters <http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ece5655_chap8.pdf>`_
            * `scipy.signal.sosfilt <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html>`_


    """

    def __init__(self, fs, columns='all', order=None, freqs=list, mode="bandpass", design="butter", pass_loss=3.0,
                 stop_atten=50.0, sos=None):
        """
        Args:
            fs (float): Nominal sampling rate of the input data.
            columns (list|"all"): Columns to apply filter on. Default: `all`.
            order (int, optional): Filter order. Default: `None`.
            freqs (list): Transition frequencies.
            mode (str): Filter mode (`lowpass`, `highpass`, `bandstop`, `bandpass`). Default: `bandpass`.
            design (str): Design of the transfert function of the filter. Default: `butter`
            pass_loss (float): Maximum attenuation in passband. Default: `3.0`.
            stop_atten (float): Minimum attenuation in stop_band. Default: `50.0`.
            sos (array, optional) : Array of second-order sections (sos) representation, must have shape (n_sections, 6). Default: `None`.
        """

        self._order = order
        self._inputfreqs = freqs
        self._mode = mode
        self._design = design
        self._pass_loss = pass_loss
        self._stop_atten = stop_atten
        self._fs = fs
        self._zi = {}
        self._sos = {}
        self._sos_custom = sos
        self._columns = columns if columns != 'all' else None

    def update(self):

        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        if self._columns is None:
            self._columns = self.i.data.columns
            for col in self._columns:
                if col not in self._sos:
                    self._sos[col] = self._design_sos()
                if col not in self._zi:
                    zi0 = signal.sosfilt_zi(self._sos[col])
                    self._zi[col] = (zi0 * self.i.data[col].values[0])
                port_o_col, self._zi[col] = signal.sosfilt(self._sos[col], self.i.data[col].values.T,
                                                           zi=self._zi[col])
                self.o.data.loc[:, col] = port_o_col

    def _design_sos(self):

        if self._sos_custom is None:
            # Calculate an IIR filter kernel for a given sampling rate.
            sos, self._freqs = construct_iir_filter(fs=self._fs, freqs=self._inputfreqs, mode=self._mode,
                                                    order=self._order, design=self._design , pass_loss=self._pass_loss, stop_atten=self._stop_atten)
            return sos
        else:
            if self._sos_custom.shape[1] == 6:
                return self._sos_custom
            else:
                raise ValueError("sos must have shape (n_sections, 6) ")


class FIRFilter(Node):
    """Apply FIR filter to signal.

    If ``coeffs`` is `None`, this node uses adapted methods from *mne.filters* to design the filter coefficients based on the specified parameters.
    If no transition band is given, default is to use:

    * l_trans_bandwidth =  min(max(l_freq * 0.25, 2), l_freq)
    * h_trans_bandwidth =   min(max(h_freq * 0.25, 2.), fs / 2. - h_freq)

    Else, it uses ``coeffs`` as filter coefficients.

    It applies the filtering to each columns in ``columns`` using `scipy.signal.lfilter` to generate the output given the input,
    hence ensures continuity  across chunk boundaries,

    The delay introduced is estimated and stored in the meta ``FIRFilter``, ``delay``.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame and meta.

    Example:
        In this exemple, we generate a signal that is the sum of two sinus with respective periods of 1kHz and 15kHz and respective amplitudes of 1 and 0.5.
        We stream this signal using the FIRFilter node, designed for lowpass filtering at cutoff frequency 6kHz, order 20.

        * ``order`` = `20`
        * ``freqs`` = `[6000, 6100]`
        * ``mode`` = `"lowpass"`

        The FIR is a linear phase filter, so it allows one to correct for the introduced delay. Here, we retrieve the input sinus of period 1kHz.
        We plot the input signal, the output signal, the corresponding offline filtering and the output signal after delay correction.

        .. image:: /../../timeflux_dsp/doc/static/image/firfilter_io.png
           :align: center

    Notes:
        The FIR filter ensures a linear phase response, but is computationnaly more costly than the IIR filter.

        The filter is initialized to have a minimal step response, but needs a "warmup" period for the filtering to be stable, leeding to small artifacts on the first few chunks.

    """

    def __init__(self, fs=64, columns='all', order=20, freqs=list, mode="bandpass",
                 design="firwin2", phase="linear", window="hamming", coeffs=None):
        """
         Args:
            fs (float): Nominal sampling rate of the input data.
            columns (list|"all", optional): Columns to apply filter on. Default: `all`.
            order (int): Filter order.
            freqs (list): Transition frequencies.
            mode (str, optional): Filter mode (`lowpass`, `highpass`, `bandstop` or `bandpass`). Default: `bandpass`.
            design (str, optional): Design of the transfert function of the filter. Default: `firwin2`.
            phase (str, optional): Phase response (`linear`, `zero`, `zero-double` or `minimum`). Default: `linear`.
            window (str, optional): The window to use in FIR design, (`hamming`, `hann`, or `blackman`). Default: `hamming`.
            coeffs (array, optional): Custom coeffs to pass as ``b`` in `signal.filter`. Default: `None`.
        """

        self._order = order
        self._freqs = freqs
        self._mode = mode
        self._design = design  # firwin or firwin2
        self._fs = fs
        self._columns = columns if columns != 'all' else None
        self._window = window
        self._phase = phase
        self._coeffs_custom = coeffs

        # Initialize the filter kernels and states, one per stream
        self._zi = {}  # FIR filter states, one per stream
        self._coeffs = {}  # FIR filter coeffs, one per stream
        self._delay = {}  # FIR filter delays, one per stream (average)

    def update(self):
        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        # At this point, we are sure that we have some data to process
        if self._columns is None:
            self._columns = self.i.data.columns

        for col in self._columns:
            if col not in self._coeffs:
                self._coeffs[col], self._delay[col] = self._design_filter()
            if col not in self._zi:
                zi0 = signal.lfilter_zi(self._coeffs[col], 1.0)
                self._zi[col] = (zi0 * self.i.data[col].values[0])
            port_o_col, self._zi[col] = signal.lfilter(b=self._coeffs[col], a=1.0, x=self.i.data[col].values.T,
                                                       zi=self._zi[col])
            self.o.meta = {"FIRFilter": {"delay": self._delay}}
            self.o.data.loc[:, col] = port_o_col

    def _design_filter(self):
        # Calculate an FIR filter kernel for a given sampling rate.
        nyq = self._fs / 2.0

        if self._coeffs_custom is None:
            self._freqs, gains, _, _ = design_edges(freqs=self._freqs, nyq=nyq, mode=self._mode)

            fir_coeffs = construct_fir_filter(self._fs, self._freqs, gains, self._order, self._phase, self._window,
                                              self._design)
        else:
            fir_coeffs = self._coeffs_custom
        warmup = self._order - 1
        fir_delay = (warmup / 2) / self._fs
        return fir_coeffs, fir_delay
