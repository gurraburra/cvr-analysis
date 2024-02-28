from process_control import ProcessNode
import matplotlib.pyplot as plt
import numpy as np

def plotCorrelation(series_a, series_b, timeshift, ax = None, tr = 1):
    if ax is None:
        ax = plt

    indices_a = np.arange(len(series_a))
    ax.plot(indices_a * tr, (series_a - series_a.mean()) / series_a.std(), label = "series_a")
    
    indices_b = np.arange(timeshift, len(series_b) + timeshift)
    ax.plot(indices_b * tr, (series_b - series_b.mean()) / series_b.std(), label = "series_b")


class CVRStat(ProcessNode):
    outputs = tuple()
    def _run(self, bold_data, regressor, global_series, global_timeshift, cvr_timeshift_list, cvr_amplitude_list) -> tuple:
        return tuple()
    
