"""Tests for decimation nodes"""
import numpy as np
import pandas as pd
import pytest
from timeflux.helpers.testing import DummyData, Looper

from timeflux_dsp.nodes.filters import DropRows, Resample


@pytest.fixture(scope="module")
def generator(rate=10, jitter=0.05):
    """Create object to mimic data streaming """
    generator = DummyData(rate=rate, jitter=jitter)
    return generator


def test_droprows_1(generator):
    """ Test DropRows
    Test that the node correctly maintains an internal state, when size of
    chunk is not a multiple of factor
    """
    factor = 2
    generator.reset()
    node = DropRows(factor=factor, method=None)
    node.i.data = generator.next(5)
    node.update()
    expected_o = generator._data.iloc[[1, 3]]
    expected_p = generator._data.iloc[[4]]
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)


def test_droprows_2(generator):
    """ Test DropRows
    test for factor = 2, 3, 4, 8 that the output is equivalent to applying a
    rolling window and taking the mean over the samples.
    size of chunk is 5 rows.
    """

    for factor in [2, 3, 4, 8]:
        generator.reset()
        node = DropRows(factor=factor, method="mean")
        looper = Looper(node=node, generator=generator)
        out_data, _ = looper.run(chunk_size=10)
        expected = (
            generator._data.rolling(window=factor, min_periods=factor, center=False)
            .mean()
            .iloc[np.arange(factor - 1, len(generator._data), factor)]
        )
        pd.testing.assert_frame_equal(
            out_data.iloc[: len(generator._data) // factor], expected
        )


def test_resample(generator):
    """ Test Resample
    """
    factor = 2

    generator.reset()
    node = Resample(factor=factor)
    chunk = generator.next(5)

    expected_o = pd.DataFrame(
        data=np.array(
            [
                [0.47955525, 0.37772175, 0.77298, 0.830601, 0.55936975],
                [0.51563125, 0.82348875, 0.844093, 0.137803, 0.45838375],
            ]
        ),
        index=generator._data.index[[0, 2]],
    )
    expected_p = generator._data.iloc[[4]]

    node.i.data = chunk
    node.update()
    pd.testing.assert_frame_equal(node.o.data, expected_o)
    pd.testing.assert_frame_equal(node._previous, expected_p)
