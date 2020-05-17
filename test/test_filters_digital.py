"""Tests for iir and fir nodes"""

import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
from timeflux.helpers.testing import ReadData, Looper
from timeflux_dsp.nodes.filters import FIRFilter, IIRFilter


@pytest.fixture(scope="module")
def rate():
    return 50


# ------------------------------------------------------------------
# Create a signal for demonstration of IIR/FIR filtering .
# ------------------------------------------------------------------
@pytest.fixture(scope="module")
def generator(rate):
    """Create object to mimic data streaming """
    # Signal of 300 points (sum of two sinus at 0.5 Hz  and 10 Hz) sampled at 50 kHz.

    n = int(rate * 30)  # 30 seconds of data
    m = 3

    f_carrier = 0.5
    a_carrier = 1.0

    f_noise = 10
    a_noise = 0.1

    tm.K = m  # tm will generate 7 columns
    original = tm.makeTimeDataFrame(n, freq="L").rename(
        columns={"A": "noise", "B": "carrier", "C": "signal",}
    )
    original.index = original.index[0] + pd.to_timedelta(np.arange(n) / rate, unit="s")
    t = (original.index - original.index[0]) / np.timedelta64(1, "s")
    original["noise"] = a_noise * np.sin(2 * np.pi * f_noise * t)
    original["carrier"] = a_carrier * np.sin(2 * np.pi * f_carrier * t)
    original["signal"] = original.carrier + original.noise
    data = ReadData(original[["signal"]])
    data._rate = rate
    return data


# -----------------------------------------------------------------------------
# Test that filtering online (chunk by chunk) is the same as filtering offline
# -----------------------------------------------------------------------------
def test_cascade_iirfilter(generator):
    """ Test IIRFilter cascade  """
    rate = generator._rate
    cutoff_hz = 3
    # create filter
    node_iir = IIRFilter(
        rate=rate, frequencies=[cutoff_hz], filter_type="lowpass", order=3
    )

    # Filter online (chunk by chunk)
    # --------------
    # reset the data streamer
    generator.reset()
    looper = Looper(node=node_iir, generator=generator)
    cascade_output, _ = looper.run(chunk_size=5)

    # Filter offline (whole data)
    # --------------
    # reset the data streamer
    generator.reset()
    looper = Looper(node=node_iir, generator=generator)
    continuous_output, _ = looper.run(chunk_size=None)

    # assert filters coeffs are correct
    expected_sos = np.array(
        [
            [0.00475052, 0.00950105, 0.00475052, 1.0, -0.6795993, 0.0],
            [1.0, 1.0, 0.0, 1.0, -1.57048578, 0.68910035],
        ]
    )
    np.testing.assert_array_almost_equal(node_iir._sos, expected_sos)

    # assert signal filtered offline and online are the same after the warmup period.
    order = 3
    warmup = 100 * (order) / (node_iir._rate)
    np.testing.assert_array_almost_equal(
        continuous_output.iloc[int(warmup * node_iir._rate) :].values,
        cascade_output.iloc[int(warmup * node_iir._rate) :].values,
        3,
    )


def test_cascade_firfilter(generator):
    """ Test FIRFilter"""
    rate = generator._rate

    # create the filter
    node_fir = FIRFilter(
        rate=rate, columns="all", order=20, frequencies=[3, 4], filter_type="lowpass"
    )
    expected_coeffs = np.array(
        [
            -0.00217066,
            -0.00208553,
            -0.00108039,
            0.00392436,
            0.01613796,
            0.03711417,
            0.06535715,
            0.09608169,
            0.12241194,
            0.13763991,
            0.13763991,
            0.12241194,
            0.09608169,
            0.06535715,
            0.03711417,
            0.01613796,
            0.00392436,
            -0.00108039,
            -0.00208553,
            -0.00217066,
        ]
    )

    # Filter online (chunk by chunk)
    # --------------
    # reset the data streamer
    generator.reset()
    looper = Looper(node=node_fir, generator=generator)
    cascade_output, cascade_meta = looper.run(chunk_size=5)

    # Filter offline (whole data)
    # --------------
    # reset the data streamer
    generator.reset()
    looper = Looper(node=node_fir, generator=generator)
    cascade_output, metas = looper.run(chunk_size=None)

    # Filter offline
    # --------------
    node_fir.i.data = generator._data.copy()
    node_fir.update()
    continuous_output = node_fir.o.data

    delay = cascade_meta[0]["delay"]

    # assert filters coeffs are correct
    np.testing.assert_array_almost_equal(node_fir._coeffs, expected_coeffs)

    # assert signal filtered offline and online are the same
    warmup = delay * 2
    pd.testing.assert_frame_equal(
        continuous_output.iloc[int(warmup * node_fir._rate) :],
        cascade_output.iloc[int(warmup * node_fir._rate) :],
        check_less_precise=3,
    )

    # correct for induced delay
    fir_o_delayed = cascade_output.copy()
    fir_o_delayed.index -= delay * np.timedelta64(1, "s")


def test_bandpass_power_ratio():
    # Prepare a reproducible example
    np.random.seed(42)
    rate = 512
    n = int(rate * 30)  # 30 seconds of data
    lo, hi = 10, 20
    original = tm.makeTimeDataFrame(n)
    original.index = original.index[0] + pd.to_timedelta(np.arange(n) / rate, unit="s")

    node_iir = IIRFilter(rate=rate, frequencies=[lo, hi], filter_type="bandpass")
    node_fir = FIRFilter(
        rate=rate, frequencies=[lo, hi], filter_type="bandpass", order=rate
    )

    for node in [node_iir, node_fir]:
        node.i.data = original.copy()
        node.update()
        filtered = node.o.data

        freqs = np.fft.fftfreq(n, 1 / rate)

        # Compare in frequency domain
        x = original.values
        X = np.fft.fft(x, axis=0)
        y = filtered.values
        Y = np.fft.fft(y, axis=0)

        f_tol = 1.5  # frequency tolerance from mne
        inner_mask = (freqs > (lo + f_tol)) & (freqs < (hi - f_tol))
        outer_mask = (freqs < (lo - f_tol)) | (freqs > (hi + f_tol))
        pass_ratio = np.abs(Y[inner_mask]) / np.abs(X[inner_mask])
        block_ratio = np.abs(Y[outer_mask]) / np.abs(X[outer_mask])

        # Note for the asserts: absolute tolerances taken from mne would be
        # 0.02 for pass and 0.20 for blocked.
        # We are currently not as good so we are being a bit more relaxed (0.05)
        # for the pass ratio
        np.testing.assert_allclose(np.mean(pass_ratio, axis=0), 1, atol=0.05)
        np.testing.assert_allclose(np.mean(block_ratio, axis=0), 0, atol=0.20)


def test_custom_sos(generator):
    """ Test sos parameter from IIRFilter"""
    rate = generator._rate

    # sos has valid form
    sos = np.array(
        [
            [0.00475052, 0.00950105, 0.00475052, 1.0, -0.6795993, 0.0],
            [1, 1.0, 0.0, 1.0, -1.57048578, 0.68910035],
        ]
    )
    node_iir_custom1 = IIRFilter(
        rate=rate, order=None, frequencies=None, filter_type=None, sos=sos
    )
    generator.reset()
    chunk = generator.next(5).copy()
    node_iir_custom1.i.data = chunk.copy()
    node_iir_custom1.update()

    # sos does not have valid form
    with pytest.raises(ValueError):
        node_iir_custom2 = IIRFilter(
            rate=rate, order=None, frequencies=None, filter_type=None, sos=sos[:, :5]
        )
        node_iir_custom2.i.data = chunk.copy()
        node_iir_custom2.update()
