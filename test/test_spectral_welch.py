"""Tests for nodes from timeflux_dsp.nodes.spectral"""
import numpy as np
import pandas as pd
import pandas.util.testing as tm

from timeflux_dsp.nodes.spectral import Welch

# ------------------------------------------------------------------
# Create a signal for demonstration of Welch Periodogram .
# ------------------------------------------------------------------
# Here, we create three sinus with sampling rate 512 Hz, and of frequency 5, 15 and 50Hz,
# a dirac, a square, a centered dirac, mixed with white noise.
# We test if the peak frequency of the Welch node is correct.


# Prepare a reproducible example
np.random.seed(42)
rate = 512
n = rate * 10
m = 7
tm.K = m  # tm will generate 7 columns
original = tm.makeTimeDataFrame(n, freq="L").rename(
    columns={
        "A": "noise",
        "B": "dirac",
        "C": "centered dirac",
        "D": "square",
        "E": "sin5",
        "F": "sin15",
        "G": "sin50",
    }
)
original.index = original.index[0] + pd.to_timedelta(np.arange(n) / rate, unit="s")
t = (original.index - original.index[0]) / np.timedelta64(1, "s")
original["dirac"] = (original.index == original.index[0]).astype(float)
original["centered dirac"] = (original.index == original.index[n // 2]).astype(float)
original["square"] = (original.index <= original.index[rate]).astype(float)
original["sin5"] = 10 * np.sin(2 * np.pi * 5 * t) + original["noise"]
original["sin15"] = 10 * np.sin(2 * np.pi * 15 * t) + original["noise"]
original["sin50"] = 10 * np.sin(2 * np.pi * 50 * t) + original["noise"]


def test_welch():
    nperseg = 1024
    node = Welch(rate=rate, nperseg=nperseg, scaling="density")
    node.i.data = original
    node.update()

    expected_peaks = {
        "dirac": 0.0,
        "centered dirac": 1.0,
        "square": 0.5,
        "sin5": 5.0,
        "sin15": 15.0,
        "sin50": 50.0,
    }
    for space, expected_peak in expected_peaks.items():
        assert (
            node.o.data.loc[dict(space=space)]
            .frequency[node.o.data.loc[dict(space=space)].argmax("frequency")]
            .frequency.values[0]
            == expected_peak
        )
