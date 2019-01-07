"""Tests for nodes"""

import pytest
import pandas as pd
import numpy as np
from timeflux.core.registry import Registry
import helpers
import xarray as xr
from timeflux_dsp.nodes.peaks import RealTimeDetect, WindowDetect, Rate

Registry.cycle_start = 0
Registry.rate = 1


def assert_dict_almost_equal(dict1, dict2, decimal=7):
    np.testing.assert_array_equal(list(dict1.keys()), list(dict2.keys()))
    for k in dict1:
        np.testing.assert_almost_equal(dict1[k], dict2[k], decimal=decimal)


data_ppg = pd.read_csv("../test/data/test_data_ppg.csv", index_col=None)
data_ppg = pd.DataFrame(index=pd.to_datetime(data_ppg['index'].values), data = data_ppg["PPG"].values, columns=['PPG'])

data = helpers.CustomData( data=data_ppg[:100])

def test_realtimepeak():
    node = RealTimeDetect(delta=0.1, tol=0.5)
    data.reset()
    # mimic the scheduler
    output_peaks = []
    output_peaks.append(node.o.data)
    a = data.next(5).copy()
    while not a.empty:
        node.i.data = a.copy()
        node.update()
        output_peaks.append(node.o.data)
        a = data.next(5)
    cascade_output = pd.concat(output_peaks)

    expected_index = pd.DatetimeIndex(['2018-11-19 11:06:39.620900', '2018-11-19 11:06:39.794709043'], dtype='datetime64[ns]', freq=None)
    expected_labels = ['peak', 'valley']
    expected_data_peak = {'value': np.array([1.00546074]), 'lag': 0.03125, 'interval': 0.654236268}
    expected_data_valley = {'value': np.array([-1.01101112]), 'lag': 0.046875, 'interval': 0.654236268}
    pd.testing.assert_index_equal(expected_index, cascade_output.index)
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(expected_data_peak, cascade_output.data.values[0])
    assert_dict_almost_equal(expected_data_valley, cascade_output.data.values[1])


def test_windowpeak():
    data = helpers.CustomData( data=data_ppg[:200])
    node = WindowDetect(window=0.8, tol=0.5)
    data.reset()
    # mimic the scheduler
    output_peaks = []
    output_peaks.append(node.o.data)
    a = data.next(5).copy()
    while not a.empty:
        node.i.data = a.copy()
        node.update()
        output_peaks.append(node.o.data)
        a = data.next(5)
    cascade_output = pd.concat(output_peaks)
    expected_index = pd.DatetimeIndex(['2018-11-19 11:06:41.529004261', '2018-11-19 11:06:41.685242884'], dtype='datetime64[ns]', freq=None)

    expected_labels = ['peak', 'valley']
    expected_data_peak = {'value': 0.7587511539459229, 'lag': 2.562340529, 'interval': 2.562340529}
    expected_data_valley = {'value': -0.9906618595123292, 'lag': 2.718579152, 'interval': 2.718579152}
    pd.testing.assert_index_equal(expected_index, cascade_output.index)
    np.testing.assert_array_equal(expected_labels, cascade_output.label.values)
    assert_dict_almost_equal(expected_data_peak, cascade_output.data.values[0])
    assert_dict_almost_equal(expected_data_valley, cascade_output.data.values[1])


t0 = pd.Timestamp('2018-11-19 11:06:41.529004261')
expected_interval = 3
expected_rate = 1/expected_interval
t1 = t0+expected_interval*np.timedelta64(1,'s')
peak_event0 = pd.DataFrame(index=[t0], data = [['peak', {'interval':expected_interval}]], columns = ['label', 'data'])
node_rate = Rate(window=0)
node_rate.update()



# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(data._data)
# plt.plot(data_ppg.iloc[data_ppg.index.isin(cascade_output.loc[cascade_output.label =="peak"].index)] , '^')
# plt.plot(data_ppg.iloc[data_ppg.index.isin(cascade_output.loc[cascade_output.label =="valley"].index)] , 'v')
# plt.xlabel('time')
# plt.ylabel('signal')
# plt.title("WindowPeaks] PPG signal and the detected peaks and valleys")
# plt.legend(["i.data", "o.data[peak]", "o.data[valley]"], loc=1)
# plt.show()