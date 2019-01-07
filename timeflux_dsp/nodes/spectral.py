# -*- coding: utf-8 -*-

"""

This module contains nodes for spectral analysis with timeflux.

"""


import xarray as xr
import logging
from scipy.signal.spectral import fftpack

from timeflux.core.node import Node
from timeflux.helpers.clock import *


class FFT(Node):
    """
    Compute the one-dimensional discrete Fourier Transform for each column using the fast fourier tranform algorithm.

    Attributes:
        i (Port): default data input, expects DataFrame.
        o (Port): default output, provides XArray.


    Example:

            In this exemple, we simulate a white noise and we apply FFT:
            * fs = 10.O
            * nfft = 5
            * return_onesided = False

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


    Note:
       This node should be used after a buffer.
       For more details, see documentation from  https://github.com/scipy/scipy/tree/master/scipy/fftpack
    """
    def __init__(self, fs=1.0,  nfft=None, return_onesided=True):
        """
               Initialize the node.

               Args:
                   fs (float): Nominal sampling rate of the input data.
                   nfft (int, None): Length of the Fourier transform. The default is the length of the chunk.
                   return_onesided (bool, True): If `True`, return a one-sided spectrum for real data.
                                                 If`False` return a two-sided spectrum.
                                                 (Note that for complex data, a two-sided spectrum is always returned.)

        """

        self._fs = fs
        self._nfft = nfft
        if return_onesided:
            self._sides = 'onesided'
        else:
            self._sides = 'twosided'
        if self._nfft is not None:
            self._set_freqs()

    def _check_nfft(self):

        # Check validity of nfft at first chunk
        if self._nfft is None:
            logging.info("nfft := length of the chunk ")
            self._nfft = len(self.i.data)
            self._set_freqs()
        elif self._nfft < len(self.i.data):
            raise ValueError('nfft must be greater than or equal to length of chunk.')
        else:
            self._nfft = int(self._nfft)

    def _set_freqs(self):

        # Set freqs indexes
        if self._sides == 'onesided':
            self._freqs = np.fft.rfftfreq(self._nfft, 1 / self._fs)
        else:
            self._freqs = fftpack.fftfreq(self._nfft, 1 / self._fs)

    def update(self):

        self.o = self.i
        if self.o.data is not None:
            if not self.o.data.empty:
                self._check_nfft()
                self.o.data = self.i.data
                if self._sides == 'twosided':
                    func = fftpack.fft
                else:
                    self.o.data = self.o.data.apply(lambda x: x.real)
                    func = np.fft.rfft
                values =  func(self.o.data.values.T, n=self._nfft).T
                ## deprecated MultiIndex --> XArray
                # self.o.data = pd.DataFrame(index = pd.MultiIndex.from_product([[self.o.data.index[-1]], self._freqs], names = ["times", "freqs"]), data = values, columns = self.o.data.columns)
                self.o.data = xr.DataArray(np.stack([values], 0),
                                    coords=[[self.o.data.index[-1]], self._freqs, self.o.data.columns],
                                    dims=['times', 'freqs', 'space'])
