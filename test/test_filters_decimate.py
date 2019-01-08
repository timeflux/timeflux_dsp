"""Tests for decimation nodes"""

import pandas as pd
import numpy as np
from timeflux.core.registry import Registry

import helpers

from timeflux_dsp.nodes.filters import DropRows, Resample

Registry.cycle_start = 0
Registry.rate = 1



fs=10
data = helpers.DummyData( rate=fs, jitter=.05,)
all_data = data._data

def test_droprows_1():
    # test that the node correctly maintains an internal state, when size of chunk is not a multiple of factor

    factor = 2
    node = DropRows(factor=factor, method=None)
    node.i.data = data.next(5)
    node.update()
    expected_o = all_data.iloc[[1,3]]
    expected_p = all_data.iloc[[4]]
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)

def test_droprows_2():
    # test for factor = 2, 3, 4, 8 that the output is equivalent to applying a rolling window and taking the mean over the samples.
    # size of chunk is 5 rows.

    for factor in [2,3,4,8]:
        data.reset()
        node = DropRows(factor=factor, method="mean")
        a = data.next(5)
        port_o = []
        while not a.empty:
            node.i.data = a
            node.update()
            port_o.append(node.o.data)
            a = data.next(5)
        out_data = pd.concat(port_o)

        expected = all_data.rolling(window=factor, min_periods=factor, center=False).mean().iloc[np.arange(factor-1,len(all_data), factor)]
        pd.testing.assert_frame_equal(out_data, expected)
        assert len(out_data)==len(all_data)//factor


def test_resample():
    factor = 2
    data.reset()
    node = Resample(factor=factor)
    a = data.next(5)
    port_o = []

    expected_o = pd.DataFrame(data=np.array([[0.47955525, 0.37772175, 0.77298   , 0.830601  , 0.55936975],
                                             [0.51563125, 0.82348875, 0.844093  , 0.137803  , 0.45838375]]),
                              index = all_data.index[[0,2]])
    expected_p = all_data.iloc[[4]]

    node.i.data = a
    node.update()
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)