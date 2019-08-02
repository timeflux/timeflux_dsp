"""Tests for detection peaks nodes"""
from numbers import Number
import numpy as np
import pandas as pd
import pandas.util.testing as tm
import pytest
from timeflux.helpers.tests import CustomData

from timeflux_dsp.nodes.peaks import LocalDetect, RollingDetect


@pytest.fixture(scope='module')
def data_ppg():
    """Create object to mimic data streaming """
    # Signal of 300 points (sum of two sinus at 0.5 Hz  and 10 Hz) sampled at 50 kHz.
    df = pd.read_csv("../test/data/test_data_ppg.csv", index_col=None)
    df = pd.DataFrame(index=pd.to_datetime(df['index'].values), data=df["PPG"].values, columns=['PPG'])
    data_ppg = CustomData(df[:200])
    return data_ppg


def assert_dict_almost_equal(dict1, dict2, decimal=7):
    np.testing.assert_array_equal(list(dict1.keys()), list(dict2.keys()))
    for k in dict1:
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
    original = tm.makeTimeDataFrame(num_points, freq='L')
    original.index = original.index[0] + pd.to_timedelta(np.arange(num_points) / rate, unit='s')

    sigmas = [5.0, 1.0]
    lags = [5.0 / rate, 1.0 / rate]
    center_locs = [1 * rate, 5 * rate]
    intervals = [1.0, 5.0 - 1.0]

    gaussians_values = _gen_gaussians(center_locs, sigmas, num_points)
    peak_times = original.index[center_locs]

    original['A'] = gaussians_values

    data_gaussians = CustomData(original)
    node = LocalDetect(delta=0.5, tol=0.5)
    # mimic the scheduler
    output_peaks = []
    output_peaks.append(node.o.data)
    chunk = data_gaussians.next(5).copy()
    while not chunk.empty:
        node.i.data = chunk.copy()
        node.update()
        output_peaks.append(node.o.data)
        chunk = data_gaussians.next(5)
    cascade_output = pd.concat(output_peaks)
    pd.testing.assert_index_equal(peak_times,
                                  cascade_output[cascade_output.label == 'peak'].index)
    cascade_peaks = pd.concat(
        [pd.DataFrame(meta) for meta in cascade_output[cascade_output.label == 'peak'].data.values],
        ignore_index=True)

    expected_peaks = pd.DataFrame(dict(column_name=['A', 'A'],
                                       value=[1.0, 1.0],
                                       interval=intervals,
                                       lag=lags))
    pd.testing.assert_frame_equal(cascade_peaks, expected_peaks, check_like=True)


def test_localdetect_on_ppg(data_ppg):
    node = LocalDetect(delta=0.1, tol=0.5)
    data_ppg.reset()
    # mimic the scheduler
    output_peaks = []
    output_peaks.append(node.o.data)
    a = data_ppg.next(5).copy()
    while not a.empty:
        node.i.data = a.copy()
        node.update()
        output_peaks.append(node.o.data)
        a = data_ppg.next(5)
    cascade_output = pd.concat(output_peaks)

    expected_index = pd.DatetimeIndex(['2018-11-19 11:06:39.620900',
                                       '2018-11-19 11:06:39.794709043',
                                       '2018-11-19 11:06:40.605209027',
                                       '2018-11-19 11:06:40.761455675',
                                       '2018-11-19 11:06:41.560254261',
                                       '2018-11-19 11:06:41.714533810'], dtype='datetime64[ns]', freq=None)
    expected_labels = ['peak', 'valley'] * 3
    expected_data_peak = {'value': np.array([1.00546074]), 'lag': 0.03125, 'interval': 0.654236268,
                          'column_name': 'PPG'}
    expected_data_valley = {'value': np.array([-1.01101112]), 'lag': 0.046875, 'interval': 0.654236268,
                            'column_name': 'PPG'}

    pd.testing.assert_index_equal(expected_index, cascade_output.index)
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(expected_data_peak, cascade_output.data.values[0])
    assert_dict_almost_equal(expected_data_valley, cascade_output.data.values[1])


def test_rollingdetect(data_ppg):
    node = RollingDetect(window=0.8, tol=0.5)
    data_ppg.reset()
    # mimic the scheduler
    output_peaks = []
    output_peaks.append(node.o.data)
    chunk = data_ppg.next(5).copy()
    while not chunk.empty:
        node.i.data = chunk.copy()
        node.update()
        output_peaks.append(node.o.data)
        chunk = data_ppg.next(5)
    cascade_output = pd.concat(output_peaks)
    expected_index = pd.DatetimeIndex(['2018-11-19 11:06:41.529004261', '2018-11-19 11:06:41.685242884'],
                                      dtype='datetime64[ns]', freq=None)

    expected_labels = ['peak', 'valley']
    expected_data_peak = {'value': 0.7587511539459229, 'lag': 0.0, 'interval': 2.562340529, 'column_name': 'PPG'}
    expected_data_valley = {'value': -0.9906618595123292, 'lag': 2.718579152, 'interval': 2.718579152,
                            'column_name': 'PPG'}
    pd.testing.assert_index_equal(expected_index, cascade_output.index)
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(expected_data_peak, cascade_output.data.values[0])
    assert_dict_almost_equal(expected_data_valley, cascade_output.data.values[1])
