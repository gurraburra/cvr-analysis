from process_control import ProcessNode
import numpy as np
import pandas as pd 
from nilearn import glm, signal
from collections.abc import Iterable
from scipy.signal import butter, sosfiltfilt

class NewSampleTime(ProcessNode):
    outputs = ("up_sampling_factor", "new_sample_time")

    def _run(self, old_sample_time, min_sample_freq = None):
        if min_sample_freq is None:
            return 1, old_sample_time
        else:
            up_sampling_factor = max(1, int(np.ceil(min_sample_freq * old_sample_time)))
            new_sample_time = old_sample_time / up_sampling_factor
            return up_sampling_factor, new_sample_time

class ResampleTimeSeries(ProcessNode):
    """
    Resamples timeseries to new time interval
    """
    outputs = ("resampled_times","resampled_timeseries") 

    def _run(self, times : np.ndarray, timeseries : np.ndarray, sample_time : float, start_time : float = None, end_time : float = None) -> tuple:
        if start_time is None:
            start_time = np.min(times)
        if end_time is None:
            end_time = np.max(times)

        t_new = np.arange(start_time, end_time + sample_time, sample_time)

        intp_func = lambda timeseries : np.interp(t_new, times[~np.isnan(timeseries)], timeseries[~np.isnan(timeseries)])

        if isinstance(timeseries, np.ndarray) and (timeseries.ndim == 1 or timeseries.ndim == 2):
            if timeseries.ndim == 2:
                resampled_timeseries = np.apply_along_axis(intp_func, 0, timeseries)
            else:
                resampled_timeseries = intp_func(timeseries)
        elif isinstance(timeseries, pd.DataFrame):
            resampled_timeseries = timeseries.apply(intp_func, 0)
        else:
            raise ValueError(f"timeseries must be 1D/2D numpy array or pandas dataframe, '{type(timeseries)}' was given")

        return t_new, resampled_timeseries
    
class DetrendTimeSeries(ProcessNode):
    """
    Detrend timeseries which is assumed to be evenly sampled 
    """
    outputs = ("detrended_timeseries", )

    def _run(self, timeseries : np.ndarray, detrend_order : int = 1) -> tuple:
        # check detrend order
        if detrend_order is None:
            return timeseries, 
        else:
            # create linear confounds
            linear_confounds = glm.first_level.make_first_level_design_matrix(np.arange(timeseries.shape[0]), drift_model = "polynomial", drift_order = detrend_order).drop(columns="constant")

            if isinstance(timeseries, np.ndarray) and (timeseries.ndim == 1 or timeseries.ndim == 2):
                if timeseries.ndim == 2:
                    detrended_timeseries = signal.clean(timeseries, confounds=linear_confounds, detrend=False, standardize=False)
                else:
                    detrended_timeseries = signal.clean(timeseries[:,None], confounds=linear_confounds, detrend=False, standardize=False)[:,0]
            elif isinstance(timeseries, pd.DataFrame):
                detrended_timeseries = pd.DataFrame(signal.clean(timeseries.to_numpy(), confounds=linear_confounds, detrend=False, standardize=False), 
                                                    index = timeseries.index, columns = timeseries.columns)
            else:
                raise ValueError(f"timeseries must be 1D/2D numpy array or pandas dataframe, '{type(timeseries)}' was given")

            return detrended_timeseries, 

class TemporalFilterTimeSeries(ProcessNode):
    outputs = ("temporal_filtered_timeseries", )

    def _run(self, sample_time : float, timeseries : np.ndarray, filter_freq : float = None, filter_order : int = 6) -> tuple:
        # check if filter freq is none, return directly
        if filter_freq is None:
            return timeseries,
        else:
            # otherwise filter data
            if isinstance(filter_freq, Iterable):
                if len(filter_freq) != 2:
                    raise ValueError("'filter_freq' can either be a single value for lowpass filter of list of two values for bandpass filter.")
                btype = "bandpass"
            else:
                btype = "lowpass"

            sos = butter(filter_order, filter_freq, btype, fs = 1 / sample_time, output="sos")

            filt_func = lambda x : sosfiltfilt(sos, x, axis = 0)

            if isinstance(timeseries, np.ndarray) and (timeseries.ndim == 1 or timeseries.ndim == 2):
                temporal_filtered_timeseries = filt_func(timeseries)
            elif isinstance(timeseries, pd.DataFrame):
                temporal_filtered_timeseries = timeseries.apply(filt_func, 0)
            else:
                raise ValueError(f"timeseries must be 1D/2D numpy array or pandas dataframe, '{type(timeseries)}' was given")

            return temporal_filtered_timeseries, 

class DownsampleTimeSeries(ProcessNode):
    """
    Downsamples data.
    """
    outputs = ("down_sampled_timeseries",)

    def _run(self, down_sampling_factor : int, timeseries : np.ndarray) -> tuple:
        return timeseries[::down_sampling_factor], 

    
