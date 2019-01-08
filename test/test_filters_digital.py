"""Tests for iir and fir nodes"""

import pytest
import pandas as pd
import numpy as np
from timeflux.core.registry import Registry

import helpers
from copy import copy

from timeflux_dsp.nodes.filters import FIRFilter, IIRFilter

Registry.cycle_start = 0
Registry.rate = 1

#------------------------------------------------------------------
# Create a signal for demonstration of IIR/FIR/FiltFilt filtering .
#------------------------------------------------------------------
# 320 samples of (1000Hz + 15000 Hz) at 48 kHz
sample_rate = 48*1e3
nsamples = 320

F_1KHz = 1*1e3
A_1KHz = 1.0

F_15KHz = 15*1e3
A_15KHz = 0.5

# The cutoff frequency of the filter: 6KHz
cutoff_hz = 6*1e3

t = np.arange(nsamples) / sample_rate
signal = A_1KHz * np.sin(2*np.pi*F_1KHz*t)
noise = A_15KHz*np.sin(2*np.pi*F_15KHz*t)

data_clean = pd.DataFrame(data=copy(signal), index = t)
data_noise = pd.DataFrame(data=copy(noise), index = t)

data_input = data_clean + data_noise

custom_data = helpers.CustomData(data_input)

#------------------------------------------------------------------
# Create a IIR filter and apply it to signal.
#------------------------------------------------------------------
# create filter
node_iir = IIRFilter(fs = sample_rate, columns = 'all', order = 3, freqs = [cutoff_hz], mode="lowpass")

expected_sos = np.array([[ 0.03168934,  0.06337869,  0.03168934,  1.        , -0.41421356,  0.        ],
                         [ 1.        ,  1.        ,  0.        ,  1.        , -1.0448155 ,  0.47759225]])



def test_cascade_iirfilter():
    # reset the data streamer
    custom_data.reset()

    output_iir = []
    a = custom_data.next(5)
    node_iir.i.data = a.copy()
    node_iir.update()

    # mimic the scheduler
    output_iir.append(node_iir.o.data)
    a = custom_data.next(5).copy()
    while not a.empty:
        node_iir.i.data = a.copy()
        node_iir.update()
        output_iir.append(node_iir.o.data)
        a = custom_data.next(5)
    cascade_output = pd.concat(output_iir)

    node_iir.i.data = data_input.copy()
    node_iir.update()
    continuous_output = node_iir.o.data

    # assert filters coeffs are correct
    np.testing.assert_array_almost_equal(node_iir._sos[0], expected_sos)

    # assert signal filtered offline and online are the same after the warmup period.
    order = 3
    warmup = 10 * (order) / (node_iir._fs)
    np.testing.assert_array_almost_equal(continuous_output.iloc[int(warmup * node_iir._fs):].values,
                                         cascade_output.iloc[int(warmup * node_iir._fs):].values, 3)



def test_custom_sos():
    # sos has valid form
    node_iir_custom1 = IIRFilter(fs=sample_rate, columns='all', order=None, freqs=None, mode=None, sos=expected_sos)
    custom_data.reset()
    a = custom_data.next(5).copy()
    node_iir_custom1.i.data = a.copy()
    node_iir_custom1.update()

    # sos does not have valid form
    from timeflux.core.exceptions import TimefluxException
    with pytest.raises(ValueError):
        node_iir_custom2 = IIRFilter(fs=sample_rate, columns='all', order=None, freqs=None, mode=None,
                                     sos=expected_sos[:, :5])
        node_iir_custom2.i.data = a.copy()
        node_iir_custom2.update()

#------------------------------------------------------------------
#  Create a FIR filter and apply it to signal.
#------------------------------------------------------------------

# create the filter
node_fir = FIRFilter(fs=sample_rate, columns="all", order=20, freqs=[cutoff_hz, cutoff_hz+1], mode="lowpass")

def test_cascade_firfilter():
    # reset the data streamer
    custom_data.reset()

    # mimic the scheduler
    output_fir = []
    output_delay = []
    a = custom_data.next(5).copy()
    while not a.empty:
        node_fir.i.data = a
        node_fir.update()
        output_fir.append(node_fir.o.data)
        a = custom_data.next(5).copy()
        delay = node_fir.o.meta['FIRFilter']['delay'][0]
        output_delay.append(delay)

    cascade_output = pd.concat(output_fir)

    node_fir.i.data = data_input.copy()
    node_fir.update()
    continuous_output = node_fir.o.data

    delay = node_fir.o.meta['FIRFilter']['delay'][0]

    # assert filters coeffs are correct
    expected_coeffs = np.array([2.77264604e-03, 2.93000384e-03, -1.88582923e-04, -1.08779615e-02,
                                -2.48315398e-02, -2.37339883e-02, 1.45134845e-02, 9.45743397e-02,
                                1.90496858e-01, 2.56391830e-01, 2.56391830e-01, 1.90496858e-01,
                                9.45743397e-02, 1.45134845e-02, -2.37339883e-02, -2.48315398e-02,
                                -1.08779615e-02, -1.88582923e-04, 2.93000384e-03, 2.77264604e-03])

    np.testing.assert_array_almost_equal(node_fir._coeffs[0], expected_coeffs)

    # assert signal filtered offline and online are the same
    warmup = delay * 2
    pd.testing.assert_frame_equal(continuous_output.iloc[int(warmup*node_fir._fs):], cascade_output.iloc[int(warmup*node_fir._fs):], check_less_precise=3)


    # correct for induced delay
    fir_o_delayed = cascade_output.copy()
    fir_o_delayed.index -= delay
