from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class iCO2Stim:
    def __init__(self, durations : list[float], values : list[float], name : str = "") -> None:
        assert len(durations) == len(values), "Durations and values needs to have same length."
        self._durations = durations
        self._values = values
        self._name = name
        self._plot_times = self.plotTimes(self.durations)
        self._plot_values = self.plotValues(self.values)

    @property
    def durations(self) -> list[float]:
        return self._durations
    
    @property
    def values(self) -> list[float]:
        return self._values
    
    @property
    def name(self) -> list[float]:
        return self._name
    
    @name.setter
    def name(self, val) -> None:
        self._name = val
    
    @property
    def plot_times(self) -> list[float]:
        return self._plot_times
    
    @property
    def plot_values(self) -> list[float]:
        return self._plot_values
    
    def plot(self, ax : plt.Axes = None):
        if ax is None:
            ax = plt
        if self.name:
            ax.plot(self._plot_times, self._plot_values, label = self.name)
        else:
            ax.plot(self._plot_times, self._plot_values)

    def save(self, location : str = "", repetitions : int = 1):
        dict_ = {
                "mode" : "mr-control-box",
                "sequence" : {
                    "type": "concentration-target",
                    "durations": list(self.durations),
                    "co2": list(self.values),
                    "repetitions": repetitions
                }
            }
        with open(os.path.join(location, f"{self.name}.json"), "w") as outfile:
            json.dump(dict_, outfile, indent="\t")

    def uniformalSample(self, time_step):
        times = np.arange(self.plot_times[0], self.plot_times[-1] + time_step, time_step)
        values = np.interp(times, self.plot_times, self.plot_values)
        return times, values

    def __repr__(self):
        return f"iCO2 stimulus {self.name}\n" f"Durations: {self.durations}\n" + f"Values: {self.values}"

    @staticmethod
    def plotTimes(durations : list[float]):
        t = 0
        times = []
        for d in durations:
            times.append(t)
            t += d 
            times.append(t)
        return times
    
    @staticmethod
    def plotValues(values : list[float]):
        return np.repeat(values, 2)
    
    @staticmethod
    def combineStims(stims : list[iCO2Stim], name : str = "") -> iCO2Stim:
        durations = []
        values = []
        for stim in stims:
            durations.extend(stim.durations)
            values.extend(stim.values)
        return iCO2Stim(durations, values, name)

class iCO2Block(iCO2Stim):
    def __init__(self, duration : float, value : float) -> None:
        super().__init__([duration], [value])

class iCO2Ramp(iCO2Stim):
    def __init__(self, duration : float, from_value : float, to_value : float, division_size : float = 5) -> None:
        # nr divisions
        nr_div = int(round(duration / division_size, 0))
        # division time
        div_t = np.round(duration / nr_div, 4)
        # padding 
        if to_value > from_value:
            padding = div_t
        else:
            padding = 0
        # interp times
        intp_t = np.arange(padding, duration + padding / 2, div_t)
        # interpolate
        intp_v = np.round(np.interp(intp_t, [0, duration], [from_value, to_value]), 2)
        super().__init__([div_t] * nr_div, list(intp_v))

class iCO2Sinusoidal(iCO2Stim):
    def __init__(self, duration : float, amplitude : float, baseline : float = 0, quarter_period : bool = False, phase_offset : float = 0, division_size : float = 5) -> None:
        # nr divisions
        nr_div = int(duration / division_size)
        # check for even
        if nr_div % 2 == 0:
            nr_div += 1
        # division time
        div_t = np.round(duration / nr_div, 4)
        # padding 
        padding = div_t / 2
        # interp times
        intp_t = np.arange(padding, duration, div_t)
        # period time 
        if quarter_period:
            period_time = 4 * duration
        else:
            period_time = 2 * duration
        # interpolate
        intp_v = np.round(amplitude * np.sin(2*np.pi*intp_t / period_time + phase_offset ), 2) + baseline
        super().__init__([div_t] * int(nr_div), list(intp_v))
    