"""Tests for nodes"""

import pytest
import pandas as pd
from timeflux.core.registry import Registry
import helpers
import xarray as xr


Registry.cycle_start = 0
Registry.rate = 1

fs=10




from pylab import *

from timeflux.helpers.clock import float_index_to_time_index
#------------------------------------------------------------------
# Create a signal for demonstration of Welch Periodogram .
#------------------------------------------------------------------
# Here, we create three sinus with sampling rate 100 Hz, and of frequency 2, 4 and 6 Hz,
# mixed with white noise.

from timeflux_dsp.nodes.spectral import Welch

fs = 100
N = 1e5
amp = 2*np.sqrt(2)
freq1 = 2.0
freq2 = 4.0
freq3 = 6.0
noise_power = 0.001 * fs / 2

time = np.arange(N) / fs
x1 = amp*np.sin(2*np.pi*freq1*time)
x1 += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
x2 = amp*np.sin(2*np.pi*freq2*time)
x2 += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
x3 = amp*np.sin(2*np.pi*freq3*time)
x3 += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

data = pd.DataFrame(data=np.array([x1, x2, x3]).T, index = time , columns = ["a", "b", "c"])
data = float_index_to_time_index(data)
custom_data = helpers.CustomData(data)

def test_welch():
    node = Welch(fs=fs, kwargs={"nperseg": 24})
    node.i.data = custom_data.next(24*5)
    node.update()

    expected_freq = np.array([ 0.        ,  4.16666667,  8.33333333, 12.5       , 16.66666667,
           20.83333333, 25.        , 29.16666667, 33.33333333, 37.5       ,
           41.66666667, 45.83333333, 50.        ])
    expected_data = array(
          [[2.82392393e-02, 1.08738216e-01, 1.15316328e-01],
           [1.70346552e-01, 6.04870282e-01, 6.31062808e-01],
           [1.69149652e-02, 1.43949778e-01, 4.26078729e-01],
           [1.67311459e-03, 1.24411063e-03, 1.35750621e-02],
           [5.03022223e-04, 1.18516997e-03, 1.03831397e-03],
           [8.35289919e-04, 1.51165179e-03, 2.24485467e-03],
           [1.27676532e-03, 9.77846796e-04, 1.98866140e-03],
           [1.11841708e-03, 4.83583088e-04, 1.47570989e-03],
           [1.23276369e-03, 8.57973119e-04, 1.91491784e-03],
           [7.31530044e-04, 1.11935690e-03, 1.26138106e-03],
           [9.98942910e-04, 8.51922586e-04, 7.76991819e-04],
           [1.23955123e-03, 7.41251840e-04, 9.86333522e-04],
           [5.38288045e-04, 4.99933375e-04, 4.70275702e-04]])
    expected_time = [pd.Timestamp('1970-01-01 00:00:01.190000')]
    expected_space = ["a", "b", "c"]

    expected = xr.DataArray(np.stack([expected_data], 0),
                                coords=[expected_time, expected_freq, expected_space],
                                dims=['time', 'freq', 'space'])
    xr.testing.assert_allclose(node.o.data, expected, rtol=1e-06)

def test_spectral_invalid():
    custom_data.reset()
    with pytest.raises(ValueError):
        node = Welch(fs=fs, kwargs={"nperseg": 24})
        node.i.data = custom_data.next(20)
        node.update()

