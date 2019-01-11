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
sample_rate = 50
nsamples = 300

F_carrier = 0.5
A_carrier = 1.0

F_noise = 10
A_noise = 0.1

# The cutoff frequency of the filter: 6KHz
cutoff_hz = 3

t = np.arange(nsamples) / sample_rate
signal = A_carrier * np.sin(2*np.pi*F_carrier*t)
noise = A_noise*np.sin(2*np.pi*F_noise*t)

data_clean = pd.DataFrame(data=copy(signal), index = t)
data_noise = pd.DataFrame(data=copy(noise), index = t)

data_input = data_clean + data_noise

custom_data = helpers.CustomData(data_input)

#------------------------------------------------------------------
# Create a IIR filter and apply it to signal.
#------------------------------------------------------------------
# create filter
node_iir = IIRFilter(fs = sample_rate, columns = 'all', order = 3, freqs = [cutoff_hz], mode="lowpass")

expected_sos = np.array([[ 0.00475052,  0.00950105,  0.00475052,  1.        , -0.6795993 ,    0.        ],
                        [ 1.        ,  1.        ,  0.        ,  1.        , -1.57048578,  0.68910035]])



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
expected_coeffs = np.array([-0.00217066, -0.00208553, -0.00108039,  0.00392436,  0.01613796,
                            0.03711417,  0.06535715,  0.09608169,  0.12241194,  0.13763991,
                            0.13763991,  0.12241194,  0.09608169,  0.06535715,  0.03711417,
                            0.01613796,  0.00392436, -0.00108039, -0.00208553, -0.00217066])

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

    np.testing.assert_array_almost_equal(node_fir._coeffs[0], expected_coeffs)

    # assert signal filtered offline and online are the same
    warmup = delay * 2
    pd.testing.assert_frame_equal(continuous_output.iloc[int(warmup*node_fir._fs):], cascade_output.iloc[int(warmup*node_fir._fs):], check_less_precise=3)


    # correct for induced delay
    fir_o_delayed = cascade_output.copy()
    fir_o_delayed.index -= delay
