"""Tests for decimation nodes"""
import numpy as np
import pandas as pd
import pytest

import helpers
from timeflux_dsp.nodes.filters import DropRows, Resample


@pytest.fixture(scope='module')
def data(rate=10, jitter=.05):
    """Sampling rate of the test data"""
    data = helpers.DummyData(rate=rate, jitter=jitter)
    return data


def test_droprows_1(data):
    """ Test DropRows
    Test that the node correctly maintains an internal state, when size of
    chunk is not a multiple of factor
    """
    factor = 2
    data.reset()
    node = DropRows(factor=factor, method=None)
    node.i.data = data.next(5)
    node.update()
    expected_o = data._data.iloc[[1, 3]]
    expected_p = data._data.iloc[[4]]
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)


def test_droprows_2(data):
    """ Test DropRows
    test for factor = 2, 3, 4, 8 that the output is equivalent to applying a
    rolling window and taking the mean over the samples.
    size of chunk is 5 rows.
    """

    for factor in [2, 3, 4, 8]:
        data.reset()
        node = DropRows(factor=factor, method="mean")
        chunk = data.next(5)
        port_o = []
        while not chunk.empty:
            node.i.data = chunk
            node.update()
            port_o.append(node.o.data)
            chunk = data.next(5)
        out_data = pd.concat(port_o)

        expected = data._data.rolling(window=factor, min_periods=factor, center=False).mean().iloc[
            np.arange(factor - 1, len(data._data), factor)]
        pd.testing.assert_frame_equal(out_data, expected)
        assert len(out_data) == len(data._data) // factor


def test_resample(data):
    """ Test Resample
    """
    factor = 2

    data.reset()
    node = Resample(factor=factor)
    chunk = data.next(5)

    expected_o = pd.DataFrame(data=np.array([[0.47955525, 0.37772175, 0.77298, 0.830601, 0.55936975],
                                             [0.51563125, 0.82348875, 0.844093, 0.137803, 0.45838375]]),
                              index=data._data.index[[0, 2]])
    expected_p = data._data.iloc[[4]]

    node.i.data = chunk
    node.update()
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)
