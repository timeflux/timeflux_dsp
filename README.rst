Timeflux DSP plugin
===================

This plugin provides timeflux nodes and meta-nodes for real time digital signal processing of time series.

Installation
------------

First, make sure that `Timeflux <https://github.com/timeflux/timeflux>`__ is installed.

You can then install this plugin in the `timeflux` environment:

::

    $ conda activate timeflux
    $ pip install timeflux_dsp

Modules
-------

- ``filters``: contains digital filters nodes (FIR, IIR, etc.) and resampling nodes.
- ``spectral``: contains nodes for spectral analysis.
- ``peaks``: contains nodes to detect peaks on 1D data, and estimates their characteristics.