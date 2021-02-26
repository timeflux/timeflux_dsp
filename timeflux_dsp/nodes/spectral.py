"""This module contains nodes for spectral analysis with Timeflux."""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import welch
from scipy.fft import fftfreq, rfftfreq, fft, rfft

from timeflux.core.node import Node


class FFT(Node):
    """Compute the one-dimensional discrete Fourier Transform for each column using the Fast Fourier Tranform algorithm.

    Attributes:
        i (Port): default input, expects DataFrame.
        o (Port): default output, provides DataArray.

    Example:

        In this exemple, we simulate a white noise and we apply FFT:

         * ``fs`` = `10.0`
         * ``nfft`` = `5`
         * ``return_onesided`` = `False`

        self.i.data::

                                              A         B         C
            2017-12-31 23:59:59.998745401  0.185133  0.541901  0.872946
            2018-01-01 00:00:00.104507143  0.732225  0.806561  0.658783
            2018-01-01 00:00:00.202319939  0.692277  0.849196  0.249668
            2018-01-01 00:00:00.300986584  0.489425  0.221209  0.987668
            2018-01-01 00:00:00.396560186  0.944059  0.039427  0.705575


        self.o.data::

            xarray.DataArray (times: 1, freqs: 5, space: 3)
            array([[[ 3.043119+0.j      ,  2.458294+0.j      ,  3.47464 +0.j      ],
                    [-0.252884+0.082233j, -0.06265 -1.098709j,  0.29353 +0.478287j],
                    [-0.805843+0.317437j,  0.188256+0.146341j,  0.151515-0.674376j],
                    [-0.805843-0.317437j,  0.188256-0.146341j,  0.151515+0.674376j],
                    [-0.252884-0.082233j, -0.06265 +1.098709j,  0.29353 -0.478287j]]])
            Coordinates:
              * times    (times) datetime64[ns] 2018-01-01T00:00:00.396560186
              * freqs    (freqs) float64 0.0 2.0 4.0 -4.0 -2.0
              * space    (space) object 'A' 'B' 'C'

    Notes:
       This node should be used after a buffer.

    References:
        * `scipy.fft <https://docs.scipy.org/doc/scipy/reference/fft.html>`_

    """

    def __init__(self, fs=1.0, nfft=None, return_onesided=True):
        """
        Args:
            fs (float): Nominal sampling rate of the input data.
            nfft (int|None): Length of the Fourier transform. Default: length of the chunk.
            return_onesided (bool): If `True`, return a one-sided spectrum for real data.
                                          If `False` return a two-sided spectrum.
                                          (Note that for complex data, a two-sided spectrum is always returned.)
                                          Default: `True`.
        """

        self._fs = fs
        self._nfft = nfft
        if return_onesided:
            self._sides = "onesided"
        else:
            self._sides = "twosided"
        if self._nfft is not None:
            self._set_freqs()

    def _check_nfft(self):

        # Check validity of nfft at first chunk
        if self._nfft is None:
            self.logger.debug("nfft := length of the chunk ")
            self._nfft = self.i.data.shape[0]
            self._set_freqs()
        elif self._nfft < self.i.data.shape[0]:
            raise ValueError("nfft must be greater than or equal to length of chunk.")
        else:
            self._nfft = int(self._nfft)

    def _set_freqs(self):

        # Set freqs indexes
        if self._sides == "onesided":
            self._freqs = rfftfreq(self._nfft, 1 / self._fs)
        else:
            self._freqs = fftfreq(self._nfft, 1 / self._fs)

    def update(self):

        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        self._check_nfft()
        self.o.data = self.i.data
        if self._sides == "twosided":
            func = fft
        else:
            self.o.data = self.o.data.apply(lambda x: x.real)
            func = rfft
        values = func(self.o.data.values.T, n=self._nfft).T
        self.o.data = xr.DataArray(
            np.stack([values], 0),
            coords=[[self.o.data.index[-1]], self._freqs, self.o.data.columns],
            dims=["time", "freq", "space"],
        )


class Welch(Node):
    """Estimate power spectral density using Welch’s method.

    Attributes:
       i (Port): default input, expects DataFrame.
       o (Port): default output, provides DataArray with dimensions (time, freq, space).

    Example:

    In this exemple, we simulate data with noisy sinus on three sensors (columns `a`, `b`, `c`):

        * ``fs`` = `100.0`
        * ``nfft`` = `24`

    node.i.data::
        \s                       a         b         c
        1970-01-01 00:00:00.000 -0.233920 -0.343296  0.157988
        1970-01-01 00:00:00.010  0.460353  0.777296  0.957201
        1970-01-01 00:00:00.020  0.768459  1.234923  1.942190
        1970-01-01 00:00:00.030  1.255393  1.782445  2.326175
        ...                      ...       ...       ...
        1970-01-01 00:00:01.190  1.185759  2.603828  3.315607

    node.o.data::

        <xarray.DataArray (time: 1, freq: 13, space: 3)>
        array([[[2.823924e-02, 1.087382e-01, 1.153163e-01],
            [1.703466e-01, 6.048703e-01, 6.310628e-01],
            ...            ...           ...
            [9.989429e-04, 8.519226e-04, 7.769918e-04],
            [1.239551e-03, 7.412518e-04, 9.863335e-04],
            [5.382880e-04, 4.999334e-04, 4.702757e-04]]])
        Coordinates:
            * time     (time) datetime64[ns] 1970-01-01T00:00:01.190000
            * freq     (freq) float64 0.0 4.167 8.333 12.5 16.67 ... 37.5 41.67 45.83 50.0
            * space    (space) object 'a' 'b' 'c'

    Notes:

        This node should be used after a Window with the appropriate length, with regard to the parameters
        `noverlap`, `nperseg` and `nfft`.
        It should be noted that a pipeline such as {LargeWindow-Welch} is in fact equivalent to a pipeline
        {SmallWindow-FFT-LargeWindow-Average} with SmallWindow 's parameters `length` and `step` respectively
        equivalent to `nperseg` and `step` and with FFT node with same kwargs.

    """

    def __init__(self, rate=None, closed="right", **kwargs):
        """
        Args:
            rate (float|None): Nominal sampling rate of the input data. If `None`, the rate will be taken from the input meta/
            closed (str): Make the index closed on the `right`, `left` or `center`.
            kwargs:  Keyword arguments to pass to scipy.signal.welch function.
                            You can specify: window, nperseg, noverlap, nfft, detrend, return_onesided and scaling.
        """

        self._rate = rate
        self._closed = closed
        self._kwargs = kwargs
        self._set_default()

    def _set_default(self):
        # We set the default params if they are not specifies in kwargs in order to check that they are valid, in respect of the length and sampling of the input data.
        if "nperseg" not in self._kwargs.keys():
            self._kwargs["nperseg"] = 256
            self.logger.debug("nperseg := 256")
        if "nfft" not in self._kwargs.keys():
            self._kwargs["nfft"] = self._kwargs["nperseg"]
            self.logger.debug(
                "nfft := nperseg := {nperseg}".format(nperseg=self._kwargs["nperseg"])
            )
        if "noverlap" not in self._kwargs.keys():
            self._kwargs["noverlap"] = self._kwargs["nperseg"] // 2
            self.logger.debug(
                "noverlap := nperseg/2 := {noverlap}".format(
                    noverlap=self._kwargs["noverlap"]
                )
            )

    def _check_nfft(self):
        # Check validity of nfft at first chun
        if not all(
            i <= len(self.i.data)
            for i in [self._kwargs[k] for k in ["nfft", "nperseg", "noverlap"]]
        ):
            raise ValueError(
                "nfft, noverlap and nperseg must be greater than or equal to length of chunk."
            )
        else:
            self._kwargs.update(
                {
                    keyword: int(self._kwargs[keyword])
                    for keyword in ["nfft", "nperseg", "noverlap"]
                }
            )

    def update(self):
        # copy the meta
        self.o = self.i

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # Check rate
        if self._rate:
            rate = self._rate
        elif "rate" in self.i.meta:
            rate = self.i.meta["rate"]
        else:
            raise ValueError(
                "The rate was neither explicitely defined nor found in the stream meta."
            )

        # At this point, we are sure that we have some data to process
        # apply welch on the data:
        self._check_nfft()
        f, Pxx = welch(x=self.i.data, fs=rate, **self._kwargs, axis=0)

        if self._closed == "left":
            time = self.i.data.index[-1]
        elif self._closed == "center":

            def middle(a):
                return int(np.ceil(len(a) / 2)) - 1

            time = self.i.data.index[middle(self.i.data)]
        else:  # right
            time = self.i.data.index[-1]
        # f is the frequency axis and Pxx the average power of shape (Nfreqs x Nchanels)
        # we reshape Pxx to fit the ('time' x 'freq' x 'space') dimensions
        self.o.data = xr.DataArray(
            np.stack([Pxx], 0),
            coords=[[time], f, self.i.data.columns],
            dims=["time", "frequency", "space"],
        )


class Bands(Node):
    """Averages the XArray values over freq dimension according to the frequencies bands given in arguments.

    This node selects a subset of values over the chosen dimensions, averages them along this axis and convert the result into a flat dataframe.
    This node will output as many ports bands as given bands, with their respective name as suffix.

        Attributes:
            i (Port): default output, provides DataArray with 3 dimensions (time, freq, space).
            o (Port): Default output, provides DataFrame.
            o_* (Port): Dynamic outputs, provide DataFrame.

    """

    def __init__(self, bands=None, relative=False):

        """
        Args:
           bands (dict): Define the band to extract given its name and its range.
                         An output port will be created with the given names as suffix.

        """
        bands = bands or {
            "delta": [1, 4],
            "theta": [4, 8],
            "alpha": [8, 12],
            "beta": [12, 30],
        }
        self._relative = relative
        self._bands = []
        for band_name, band_range in bands.items():
            self._bands.append(
                dict(
                    port=getattr(self, "o_" + band_name),
                    slice=slice(band_range[0], band_range[1]),
                    meta={"bands": {"range": band_range, "relative": relative}},
                )
            )

    def update(self):

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # At this point, we are sure that we have some data to process
        for band in self._bands:
            # 1. select the Xarray on freq axis in the range, 2. average along freq axis
            band_power = (
                self.i.data.loc[{"frequency": band["slice"]}].sum("frequency").values
            )  # todo: sum
            if self._relative:
                tot_power = self.i.data.sum("frequency").values
                tot_power[tot_power == 0.0] = 1
                band_power /= tot_power

            band["port"].data = pd.DataFrame(
                columns=self.i.data.space.values,
                index=self.i.data.time.values,
                data=band_power,
            )
            band["port"].meta = {**(self.i.meta or {}), **band["meta"]}
