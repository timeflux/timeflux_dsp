"""Tests for detection peaks nodes"""
from numbers import Number

import numpy as np
import os
import pandas as pd
import pandas.util.testing as tm
import pytest
from timeflux.helpers.testing import ReadData, Looper

from timeflux_dsp.nodes.peaks import LocalDetect, RollingDetect


@pytest.fixture(scope="module")
def ppg_generator():
    """Create object to mimic data streaming """
    # Signal of 300 points (sum of two sinus at 0.5 Hz  and 10 Hz) sampled at 50 kHz.

    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_data_ppg.csv'), index_col=None)
    df = pd.DataFrame(
        index=pd.to_datetime(df["index"].values), data=df["PPG"].values, columns=["PPG"]
    )
    ppg_generator = ReadData(df[:200])
    return ppg_generator


def assert_dict_almost_equal(dict1, dict2, decimal=4, keys=None):
    if keys is None:
        np.testing.assert_array_equal(list(dict1.keys()), list(dict2.keys()))
    for k in keys or dict1:
        if isinstance(dict1[k], Number):
            np.testing.assert_almost_equal(dict1[k], dict2[k], decimal=decimal)


def _gen_gaussians(center_locs, sigmas, total_length):
    xdata = np.arange(0, total_length).astype(float)
    out_data = np.zeros(total_length, dtype=float)
    for ind, sigma in enumerate(sigmas):
        tmp = (xdata - center_locs[ind]) / sigma
        out_data += np.exp(-(tmp ** 2))
    return out_data


def test_localdetect_on_gaussians():
    """ Test on a sum of two gaussians if the peak center is well estimated """
    rate = 128
    num_points = 10 * rate
    tm.K = 1  # tm will generate 1 column
    original = tm.makeTimeDataFrame(num_points, freq="L")
    original.index = original.index[0] + pd.to_timedelta(
        np.arange(num_points) / rate, unit="s"
    )

    sigmas = [5.0, 1.0]
    lags = [5.0 / rate, 1.0 / rate]
    center_locs = [1 * rate, 5 * rate]
    intervals = [1.0, 5.0 - 1.0]

    gaussians_values = _gen_gaussians(center_locs, sigmas, num_points)
    peak_times = original.index[center_locs]

    original["A"] = gaussians_values

    data_gaussians = ReadData(original)
    node = LocalDetect(delta=0.5, tol=0.5)
    # loop across chunks
    looper = Looper(data_gaussians, node)
    cascade_output, _ = looper.run(chunk_size=5)
    estimation_times = [
        pd.Timestamp(event["extremum_time"])
        for event in cascade_output[cascade_output.label == "peak"].data.values
    ]

    assert estimation_times == list(peak_times)

    data_peaks = pd.concat(
        [
            pd.DataFrame(meta, index=[pd.Timestamp(meta["extremum_time"])])
            for meta in cascade_output[cascade_output.label == "peak"].data.values
        ],
        ignore_index=True,
    )

    expected_peaks = pd.DataFrame(
        dict(column_name=["A", "A"], value=[1.0, 1.0], interval=intervals, lag=lags)
    )
    pd.testing.assert_frame_equal(
        data_peaks.drop(["now", "extremum_time", "detection_time"], axis=1),
        expected_peaks,
        check_like=True,
        atol=1e-4
    )


def test_localdetect_on_ppg(ppg_generator):
    node = LocalDetect(delta=0.5, tol=0.5)
    # reset generator
    ppg_generator.reset()
    # loop across chunks
    looper = Looper(ppg_generator, node)
    cascade_output, _ = looper.run(chunk_size=5)

    expected_extremum_times = pd.DatetimeIndex(
        [
            "2018-11-19 11:06:39.62",
            "2018-11-19 11:06:39.79",
            "2018-11-19 11:06:40.61",
            "2018-11-19 11:06:40.76",
            "2018-11-19 11:06:41.56",
            "2018-11-19 11:06:41.71",
        ]
    )
    actual_extremum_times = pd.DatetimeIndex(
        [data["extremum_time"] for data in cascade_output.data.values]
    )
    expected_labels = ["peak", "valley"] * 3

    expected_data_peak_2 = {
        "value": np.array([0.95]),
        "lag": 0.048,
        "interval": 0.98,
        "column_name": "PPG",
    }
    expected_data_valley_3 = {
        "value": np.array([-1.05]),
        "lag": 0.48,
        "interval": 0.96,
        "column_name": "PPG",
    }

    pd.testing.assert_index_equal(
        expected_extremum_times, actual_extremum_times.round("10ms")
    )
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(
        expected_data_peak_2,
        cascade_output.data.values[2],
        decimal=2,
        keys=["value", "lag", "interval", "column_name"],
    )
    assert_dict_almost_equal(
        expected_data_valley_3,
        cascade_output.data.values[3],
        decimal=2,
        keys=["value", "lag", "interval", "column_name"],
    )


def test_rollingdetect(ppg_generator):
    node = RollingDetect(length=0.8, tol=0.5, rate=64)

    # reset generator
    ppg_generator.reset()
    # loop across chunks
    looper = Looper(ppg_generator, node)
    cascade_output, _ = looper.run(chunk_size=5)

    expected_index = pd.DatetimeIndex(
        [
            "2018-11-19 11:06:39.62",
            "2018-11-19 11:06:39.79",
            "2018-11-19 11:06:40.61",
            "2018-11-19 11:06:40.76",
        ]
    )

    expected_labels = ["peak", "valley", "peak", "valley"]
    expected_data_peak_2 = {
        "value": np.array([0.95]),
        "lag": 0.84,
        "interval": 0.98,
        "column_name": "PPG",
    }
    expected_data_valley_3 = {
        "value": np.array([-1.05]),
        "lag": 0.84,
        "interval": 0.96,
        "column_name": "PPG",
    }

    pd.testing.assert_index_equal(expected_index, cascade_output.index.round("10ms"))
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(
        expected_data_peak_2,
        cascade_output.data.values[2],
        decimal=2,
        keys=["value", "lag", "interval", "column_name"],
    )
    assert_dict_almost_equal(
        expected_data_valley_3,
        cascade_output.data.values[3],
        decimal=2,
        keys=["value", "lag", "interval", "column_name"],
    )
