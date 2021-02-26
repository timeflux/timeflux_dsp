import numpy as np
import pandas as pd
from neurokit import ecg_process
from timeflux.core.branch import Branch
from timeflux.core.node import Node
from timeflux.nodes.window import Window


class Discretize(Node):
    """Discretize data based on amplitude range.

    Attributes:
    i (Port): Default input, expects DataFrame.
    o (Port): Default output, provides DataFrame.

    Args:
        range (dict): Dictionary with keys are discrete value and values
        are tuple with corresponding data ranges.
        default: Default discrete value (for data that are not contained in any range)
    """

    def __init__(self, range, default=None):

        super().__init__()
        self._default = default or np.NaN
        self._range = range
        for k, range_values in self._range.items():
            for v in range_values:
                if v[0] is None:
                    v[0] = -np.inf
                if v[1] is None:
                    v[1] = np.inf

    def _discretize_sqi(self, x):

        for k, range_values in self._range.items():
            for v in range_values:
                if (x >= v[0]) & (x <= v[1]):
                    return k
        return self._default

    def update(self):

        if not self.i.ready():
            return

        self.o = self.i
        self.o.data = self.o.data.applymap(lambda x: self._discretize_sqi(x))


class ECGQuality(Window):
    """Estimate ECG Quality

    This nodes estimates ECG Quality using neurokit toolbox, by applying function
    `ecg_process` on a rolling window.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
            from the meta.
        length (float): The length of the window, in seconds.
        step (float): The sliding step, in seconds.

    """

    def __init__(self, rate, length, step):

        try:
            from neurokit.bio.bio_ecg import ecg_process
            import sklearn.externals.joblib  # Fix for bug on neurokit
        except ModuleNotFoundError:
            self.logger.error("Neurokit is not installed")

        self._rate = rate
        super().__init__(length=length, step=step)

    def update(self):

        if not self.i.ready():
            return

        # set rate from the data if it is not yet given
        if self._rate is None:
            try:
                self._rate = self.i.meta.pop("nominal_rate")
                self.logger.info(f"Nominal rate set to {self._rate}. ")
            except KeyError:
                # If there is no rate in the meta, set rate to 1.0
                self._rate = 1.0
                self.logger.warning(
                    f"Nominal rate not supplied, considering " f"1.0 Hz instead. "
                )

        # At this point, we are sure that we have some data to process
        super().update()
        # if the window output is ready, fit the scaler with its values
        if self.o.ready() and not self.o.data.dropna().empty:
            X = self.o.data.dropna().values[:, 0]
            try:
                df = ecg_process(X, sampling_rate=self._rate, hrv_features=None)["df"]
                avg_rate = df["Heart_Rate"].dropna().mean()
                if avg_rate == np.NaN:
                    avg_quality = 0.0
                else:
                    avg_quality = df["ECG_Signal_Quality"].dropna().median()
            except Exception as e:
                self.logger.debug(e)
                avg_quality = 0.0
            self.o.data = pd.DataFrame(
                columns=["ecg_quality"],
                data=[avg_quality],
                index=[self.i.data.index[-1]],
            )


class LineQuality(Branch):
    """Estimate level of line noise

    This nodes estimates LineNoise as the ratio between good power and total power on a rolling
    window.
    Good power is defined as the sum of squared samples for signal after bandpass and Notch filtering.
    Total power is defined as the sum of squared samples for signal after bandpass filtering only.

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
          from the meta.
       length (float): The length of the window, in seconds.
       step (float): The sliding step, in seconds.

    """

    def __init__(
        self,
        rate,
        range,
        window_length=3,
        window_step=0.5,
        bandpass_frequencies=(1, 65),
        line_centers=(50, 100, 150),
    ):

        super().__init__()
        graph = {
            "nodes": [
                {
                    "id": "linefilter",
                    "module": "timeflux_dsp.nodes.filters",
                    "class": "IIRLineFilter",
                    "params": {"rate": rate, "edges_center": line_centers},
                },
                {
                    "id": "bandpass",
                    "module": "timeflux_dsp.nodes.filters",
                    "class": "IIRFilter",
                    "params": {
                        "rate": rate,
                        "order": 3,
                        "frequencies": bandpass_frequencies,
                    },
                },
                {
                    "id": "square_good",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "method": "numpy.square",
                        "apply_mode": "universal",
                    },
                },
                {
                    "id": "square_total",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "method": "numpy.square",
                        "apply_mode": "universal",
                    },
                },
                {
                    "id": "window_good",
                    "module": "timeflux.nodes.window",
                    "class": "Window",
                    "params": {
                        "length": window_length,
                        "step": window_step,
                    },
                },
                {
                    "id": "window_total",
                    "module": "timeflux.nodes.window",
                    "class": "Window",
                    "params": {
                        "length": window_length,
                        "step": window_step,
                    },
                },
                {
                    "id": "sum_good",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "method": "numpy.sum",
                        "apply_mode": "reduce",
                    },
                },
                {
                    "id": "sum_total",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "module_name": "numpy.sum",
                        "apply_mode": "reduce",
                    },
                },
                {
                    "id": "divide",
                    "module": "timeflux.nodes.expression",
                    "class": "Expression",
                    "params": {
                        "expr": "i_1/i_2",
                        "eval_on": "ports",
                    },
                },
                {
                    "id": "log",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "module_name": "numpy.log",
                        "apply_mode": "universal",
                    },
                },
                {
                    "id": "discretize",
                    "module": "timeflux_dsp.nodes.quality",
                    "class": "Discretize",
                    "params": {"range": range},
                },
            ],
            "edges": [
                {"source": "bandpass", "target": "linefilter"},
                {"source": "linefilter", "target": "square_good"},
                {"source": "bandpass", "target": "square_total"},
                {"source": "square_good", "target": "window_good"},
                {"source": "square_total", "target": "window_total"},
                {"source": "window_good", "target": "sum_good"},
                {"source": "window_total", "target": "sum_total"},
                {"source": "sum_good", "target": "divide:1"},
                {"source": "sum_total", "target": "divide:2"},
                {"source": "divide", "target": "log"},
                {"source": "log", "target": "discretize"},
            ],
        }
        self.load(graph)

    def update(self):

        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return

        # copy the meta
        self.o.meta = self.i.meta

        self.set_port("bandpass", port_id="i", data=self.i.data, meta=self.i.meta)
        self.run()
        self.o = self.get_port("discretize", port_id="o")


class AmplitudeQuality(Branch):
    """Estimate discrete signal quality index based on a temporal feature from the amplitude.

    This nodes rolls a window and applies a numpy function  given by ``method``
    (eg. ptp, max, min, mean...) over rows and discretize the result based on ``range`` .

    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.
    """

    def __init__(self, range, window_length=3, window_step=0.5, method="ptp"):

        super().__init__()
        self._range = range
        for k, range_values in self._range.items():
            for v in range_values:
                if v[0] is None:
                    v[0] = -np.inf
                if v[1] is None:
                    v[1] = np.inf
        graph = {
            "nodes": [
                {
                    "id": "window",
                    "module": "timeflux.nodes.window",
                    "class": "Window",
                    "params": {"length": window_length, "step": window_step},
                },
                {
                    "id": "criteria",
                    "module": "timeflux.nodes.apply",
                    "class": "ApplyMethod",
                    "params": {
                        "method": f"numpy.{method}",
                        "apply_mode": "reduce",
                    },
                },
                {
                    "id": "discretize",
                    "module": "timeflux_dsp.nodes.quality",
                    "class": "Discretize",
                    "params": {"range": range},
                },
            ],
            "edges": [
                {"source": "window", "target": "criteria"},
                {"source": "criteria", "target": "discretize"},
            ],
        }
        self.load(graph)

    def update(self):
        # copy the meta
        self.o.meta = self.i.meta

        # When we have not received data, there is nothing to do
        if self.i.data is None or self.i.data.empty:
            return

        self.set_port("window", port_id="i", data=self.i.data, meta=self.i.meta)

        self.run()

        self.o = self.get_port("discretize", port_id="o")
