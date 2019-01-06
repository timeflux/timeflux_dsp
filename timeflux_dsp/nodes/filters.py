""" Nodes for signal filtering """

from timeflux.core.node import Node
from scipy import signal
from timeflux.helpers.clock import *
from timeflux.core.io import Port

from ..utils.filters import _construct_fir_filter, _get_com_factor, _construct_iir_filter, _design_edges


class DropRows(Node):
    """Drop Rows (decimate) signal by an integer factor.

        This node uses pandas computationally efficient functions to drop rows.
        By default, it simply transfers one row out of `factor` and drops the others.
        If `method`is "mean" (resp. "median"), it applies a rolling window of length equals `factor, computes the mean and returns one value per window.
        It maintains an internal state to ensure that every k'th sample is picked even across chunk boundaries.


    Attributes:
        i (Port): default data input, expects DataFrame.
        o (Port): default output, provides DataFrame.

    Notes:

        Note that this node is not supposed to dejitter the timestamps, so if the input chunk is not uniformly sampled , the output chunk won’t either.
        Note also that this filter does not implement any anti-aliasing filter. Hence, it is recommended to precede this node by
        a low-pass filter (e.g., FIR or IIR) which cuts out below half of the new sampling rate.


    """

    def __init__(self, factor, method=None):
        """
                Initialize the node.
                 Args:
                    factor (int): Decimation factor. Only every k'th sample will be transferred into the output.
                    method (str|None): method to use to drop rows. If None, the values are transferred as it. If "mean" (resp. "median"),
                                       the mean (resp. median) of the samples are
        """
        self._factor = factor
        self._method = method
        self._previous = pd.DataFrame()

    def update(self):

        self.o = self.i
        if self.i.data is not None:
            if not self.i.data.empty:
                if not self._previous.empty:
                    self.i.data = pd.concat([self._previous, self.i.data], axis=0)
                if len(self.i.data) % self._factor == 0:
                    self._previous = pd.DataFrame()
                else:
                    self._previous = self.i.data.iloc[(len(self.i.data) // self._factor) * self._factor:]
                    self.i.data = self.i.data.iloc[: (len(self.i.data) // self._factor) * self._factor]
                if self._method is None:
                    self.o.data = self.i.data.iloc[np.arange(self._factor - 1, len(self.i.data), self._factor)]
                else:
                    if self._method == "mean":
                        self.o.data = \
                            self.i.data.rolling(window=self._factor, min_periods=self._factor,
                                                center=False).mean().iloc[
                                np.arange(self._factor - 1, len(self.i.data), self._factor)]
                    elif self._method == "median":
                        self.o.data = \
                            self.i.data.rolling(window=self._factor, min_periods=self._factor,
                                                center=False).mean().iloc[
                                np.arange(self._factor - 1, len(self.i.data), self._factor)]


class Resample(Node):
    """Resample signal.

    This node calls scipy.signal.resample function to decimate the signal using Fourier method.

       Attributes:
        i (Port): default data input, expects DataFrame.
        o (Port): default output, provides DataFrame.

    Notes:
        This node should be used after a buffer to assure that the FFT window has always the same length.
        See documentation of scipy.signal.resample.

    Example:
        .. literalinclude:: /../test/graphs/resample.yaml
           :language: yaml
    """

    def __init__(self, factor, window=None):
        """Resample signal using scipy.signal.resample function
            This node should be used after a buffer to assure that the FFT window has always the same length
        """
        self._factor = factor
        self._window = window
        self._previous = pd.DataFrame()

    def update(self):
        self.o = self.i
        if self.i.data is not None:
            if not self.i.data.empty:
                if not self._previous.empty:
                    self.i.data = pd.concat([self._previous, self.i.data], axis=0)
                if len(self.i.data) % self._factor == 0:
                    self._previous = pd.DataFrame()
                else:
                    self._previous = self.i.data.iloc[(len(self.i.data) // self._factor) * self._factor:]
                    self.i.data = self.i.data.iloc[: (len(self.i.data) // self._factor) * self._factor]

                self.o.data = pd.DataFrame(
                    data=signal.resample(x=self.i.data.values, num=len(self.i.data) // self._factor,
                                         window=self.window),
                    index=self.i.data.index[np.arange(0, len(self.i.data), self._factor)],
                    columns=self.i.data.columns)


class IIRFilter(Node):
    """Apply IIR filter to signal.

    If ``sos`` is None, this node uses adapted methods from mne.filters to design the filter coefficients based on the specified parameters.
    If no transition band is given, default is to use:
        * ``l_freq``::   min(max(l_freq * 0.25, 2), l_freq)
        * ``h_freq``::   min(max(h_freq * 0.25, 2.), fs / 2. - h_freq)
    Else, it uses ``sos`` as filter coefficients.

    This filter ensures continuity  across chunk boundaries, using a recursive algorithm, based on a cascade of biquads filters
    (see documentation here: http://www.eas.uccs.edu/~mwickert/ece5655/lecture_notes/ece5655_chap8.pdf) and scipy.signal.sosfilt.

    Attributes:
        i (Port): default data input, expects DataFrame.
        o (Port): default output, provides DataFrame.

    Notes:
        The filter is initialized to have a minimal step response, but needs a "warmup" period for the filtering to be stable, leeding to small artifacts on the first few chunks.
        The IIR filter is faster than the FIR filter and delays the signal less but this delay is not constant and the stability not guarenteed.

    """

    def __init__(self, fs, columns='all', order=None, freqs=list, mode="bandpass", design="butter", pass_loss=3.0,
                 stop_atten=50.0, sos=None):

        """
                Initialize the node.
                 Args:
                    fs (float): Nominal sampling rate of the input data.
                    columns (list|'all'): columns to apply filter on. Default to all.
                    order (int|None): filter order
                    freqs (list): transition frequencies
                    mode (str|'bandpass'): filter mode (lowpass, highpass, bandstop, bandpass)
                    design (str|'butter'): design of the transfert function of the filter
                    pass_loss (float|3.0): maximum attenuation in passband
                    stop_atten (float|50.0): minimum attenuation in stop_band
                    sos (array|None) : Array of second-order sections (sos) representation, must have shape (n_sections, 6).
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

        self.o = self.i
        if self.i.data is not None:
            if self._columns is None:
                self._columns = self.i.data.columns
            if not self.i.data.empty:
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
            sos, self._freqs = _construct_iir_filter(fs=self._fs, freqs=self._inputfreqs, mode=self._mode,
                                                     order=self._order, design="butter", pass_loss=3.0, stop_atten=50.0)
            return sos
        else:
            if self._sos_custom.shape[1] == 6:
                return self._sos_custom
            else:
                raise ValueError("sos must have shape (n_sections, 6) ")


class FIRFilter(Node):
    """Apply FIR filter to signal.

        If `coeffs`` is None, this node uses adapted methods from mne.filters to design the filter coefficients based on the specified parameters.
        If no transition band is given, default is to use:
            * ``l_freq``::   min(max(l_freq * 0.25, 2), l_freq)
            * ``h_freq``::   min(max(h_freq * 0.25, 2.), fs / 2. - h_freq)

        Else, it uses ``coeffs`` as filter coefficients.

        It applies the filtering to each columns in ``columns`` using scipy.signal.lfilter to generate the output given the input,
        hence ensures continuity  across chunk boundaries,

        The delay introduced is estimated and stored in the meta "FIRFilter"-->"delay".

        Attributes:
        i (Port): default data input, expects DataFrame.
        o (Port): default output, provides DataFrame and meta.


    Notes:

    The FIR filter ensures a linear phase response, but is computationnaly more costly than the IIR filter.
    The filter is initialized to have a minimal step response, but needs a "warmup" period for the filtering to be stable, leeding to small artifacts on the first few chunks.

    """

    def __init__(self, fs=64, columns='all', order=20, freqs=list, mode="bandpass",
                 design="firwin2", phase="linear", window="hamming", coeffs=None):
        """
                Initialize the node.

                 Args:
                    fs (float): Nominal sampling rate of the input data.
                    columns (list|'all'): columns to apply filter on. Default to all.
                    order (int): filter order
                    freqs (list): transition frequencies
                    mode (str|'bandpass'): filter mode ("lowpass", "highpass", "bandstop" or "bandpass")
                    design (str|'firwin2'): design of the transfert function of the filter
                    phase (str|"linear"): phase response ("zero", "zero-double" or "minimum")
                    window (float|"hamming"): The window to use in FIR design, ("hamming", "hann", or "blackman".)
                    coeffs (array|None) :
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
        self.o = self.i
        if self.i.data is not None:
            if self._columns is None:
                self._columns = self.i.data.columns
            if not self.i.data.empty:
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
        """Calculate an FIR filter kernel for a given sampling rate."""
        nyq = self._fs / 2.0

        if self._coeffs_custom is None:
            self._freqs, gains, _, _ = _design_edges(freqs=self._freqs, nyq=nyq, mode=self._mode)

            fir_coeffs = _construct_fir_filter(self._fs, self._freqs, gains, self._order, self._phase, self._window,
                                               self._design)
        else:
            fir_coeffs = self._coeffs_custom
        warmup = self._order - 1
        fir_delay = (warmup / 2) / self._fs
        return fir_coeffs, fir_delay
