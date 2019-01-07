# Timeflux DSP plugin

This plugin provides timeflux nodes and meta-nodes for real time
digital signal processing of time series.

## Installation

First, make sure that [Timeflux is installed](https://github.com/timeflux/timeflux).

You can then install this plugin in the ``timeflux`` environment:

```
$ source activate timeflux
$ pip install git+https://github.com/timeflux/timeflux_dsp
```

## Modules

### filters
This module contains digital filters nodes (FIR, IIR, ...) and resampling nodes.

### spectral
This module contains nodes for spectral analysis.

### peaks
This module contains nodes to detect peaks on 1D data, and estimates their characteristics.