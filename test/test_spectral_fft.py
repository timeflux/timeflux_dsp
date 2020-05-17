"""Tests for nodes from timeflux_dsp.nodes.spectral"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from timeflux.helpers.testing import DummyData

from timeflux_dsp.nodes.spectral import FFT

fs = 10

data = DummyData(rate=fs, jitter=0.05)
all_data = data.next(50)


def test_welch():
    data.reset()

    node = FFT(fs=fs, return_onesided=False)
    node.i.data = data.next(5)

    node.update()
    expected_freqs = [0.0, 2.0, 4.0, -4.0, -2.0]
    expected_times = [pd.Timestamp("2018-01-01 00:00:00.396560186")]
    expected_data = np.array(
        [
            [
                2.687793 + 0.0j,
                2.69977 + 0.0j,
                4.158542 + 0.0j,
                2.907866 + 0.0j,
                2.979773 + 0.0j,
            ],
            [
                -0.32328042 + 0.45056971j,
                -0.09741619 - 0.84999621j,
                0.19777914 - 0.14955481j,
                0.33690762 + 1.2010184j,
                0.65131083 + 0.03780588j,
            ],
            [
                -0.55778358 - 0.64687062j,
                0.10228369 + 0.5354582j,
                -0.09468514 + 0.40190712j,
                0.03972188 - 0.40916112j,
                -0.12479483 + 0.90610597j,
            ],
            [
                -0.55778358 + 0.64687062j,
                0.10228369 - 0.5354582j,
                -0.09468514 - 0.40190712j,
                0.03972188 + 0.40916112j,
                -0.12479483 - 0.90610597j,
            ],
            [
                -0.32328042 - 0.45056971j,
                -0.09741619 + 0.84999621j,
                0.19777914 + 0.14955481j,
                0.33690762 - 1.2010184j,
                0.65131083 - 0.03780588j,
            ],
        ]
    )

    # output for "MultiIndex" soon-deprecated version
    # expected = pd.DataFrame(index = pd.MultiIndex.from_product([expected_times, expected_freqs], names = ["times", "freqs"]), data=expected_data)

    expected = xr.DataArray(
        np.stack([expected_data], 0),
        coords=[expected_times, expected_freqs, data._data.columns],
        dims=["time", "freq", "space"],
    )

    xr.testing.assert_allclose(node.o.data, expected, rtol=1e-06)


def test_fft_invalid():
    data.reset()
    with pytest.raises(ValueError):
        node = FFT(nfft=5, fs=fs, return_onesided=False)
        node.i.data = data.next(100)
        node.update()


def test_fft_nfft():
    data.reset()
    node = FFT(nfft=None, fs=fs, return_onesided=False)
    node.i.data = data.next(10)
    node.update()
    assert node._nfft == 10
