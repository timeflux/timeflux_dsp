from timeflux_dsp.nodes.spectral import AverageBands
import numpy as np
import pandas as pd
import xarray as xr



def test_averagebands():
    # here, we create an xarray with thre axis: (time, freq, space)
    # freq is (freq) [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39]
    # time is (time) datetime64[ns] 2018-01-01 ... 2018-01-01T00:00:00.400000
    # space is (space) <U6 'simple' 'double'
    # we fill the array with X(freq=f, space='simple')=f and X(freq=f, space='double')=2*f
    # hence:
    #   - by averaging in band 'alpha:=[8,12], we expect a dataframe with columns ['simple' 'double']
    # and with respective values, the average between 9 and 11 (ie. 10) on first column and the average between 9*2 and 10*2 (ie. 20) on second column.
    #   - by averaging in band 'alpha:=[12, 30], we expect a dataframe with columns ['simple' 'double']
    # and with respective values, the average between 13, 15, 17, ..., 29 (ie. 20) on first column and the double (ie. 42) on second column.

    Nt = 5
    Nf = 20
    node = AverageBands({'alpha': [8, 12], 'beta': [12, 30]})
    freq = np.arange(1, Nf * 2, 2)
    space = ["simple", "double"]
    time = pd.date_range(
        start='2018-01-01',
        periods=Nt,
        freq=pd.DateOffset(seconds=.1))
    data = np.stack([np.repeat(np.array(freq).reshape(-1, 1), Nt, axis=1).T,
                     np.repeat(np.array(freq * 2).reshape(-1, 1), Nt, axis=1).T], 2)
    node.i.data = xr.DataArray(data,
                               coords=[time, freq, space],
                               dims=['time', 'freq', 'space'])
    node.update()

    expected_data_alpha = np.array(
        [[10., 20.],
         [10., 20.],
         [10., 20.],
         [10., 20.],
         [10., 20.]])
    expected_data_beta = np.array([[21., 42.],
                                   [21., 42.],
                                   [21., 42.],
                                   [21., 42.],
                                   [21., 42.]])

    alpha_expected = pd.DataFrame(data=expected_data_alpha, columns=["simple", "double"], index=time)
    beta_expected = pd.DataFrame(data=expected_data_beta, columns=["simple", "double"], index=time)

    pd.testing.assert_frame_equal(alpha_expected, node.o_alpha.data)
    pd.testing.assert_frame_equal(beta_expected, node.o_beta.data)

    assert node.o_alpha.meta == {'AverageBands': {'range': [8, 12], 'relative': False}}
    assert node.o_beta.meta == {'AverageBands': {'range': [12, 30], 'relative': False}}


