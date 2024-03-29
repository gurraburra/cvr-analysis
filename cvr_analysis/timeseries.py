from process_control import ProcessNode
import numpy as np
from sklearn.cluster import KMeans
import scipy.signal as sc_signal
import scipy.stats as sc_stats
import pandas as pd
import re
from nilearn import masking
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
from collections.abc import Iterable

class ResampleTimeSeries(ProcessNode):
    """
    Resamples timeseries to new time interval
    """
    outputs = ("resampled_times","resampled_series") 

    def _run(self, times : np.ndarray, series : np.ndarray, sample_time : float, lower_bound : float = None, upper_bound : float = None, padding_average_time : float = 0) -> tuple:
        if lower_bound is None:
            lower_bound = np.min(times)
        if upper_bound is None:
            upper_bound = np.max(times)
        left_padding = series[np.where( ((times - lower_bound) <= padding_average_time) & ((times - lower_bound) >= 0) )[0]].mean()
        right_padding = series[np.where( ((upper_bound - times) <= padding_average_time) & ((upper_bound - times) >= 0) )[0]].mean()

        t_new = np.arange(lower_bound, upper_bound + sample_time, sample_time)
        return t_new, np.interp(t_new, times, series, left = left_padding, right = right_padding)
            

class BaselinePlateau(ProcessNode):
    """
    Finds baseline and plateoa by using a kmeans algorithm
    """
    outputs = ("baseline", "plateau")

    def _run(self, series : np.ndarray) -> tuple:
        kmeans_centers = KMeans(n_clusters = 2, n_init="auto").fit(series.reshape(-1, 1)).cluster_centers_
        return kmeans_centers.min(), kmeans_centers.max()
    
class SavgolFilter(ProcessNode):
    """
    Apply a savgol filter
    """
    outputs = ("filtered_series", )
    def _run(self, series : np.ndarray, sample_time : float, filter_size : float, polynom_order : int, deriv : int = 0) -> tuple:
        filtered = sc_signal.savgol_filter(series, max(int(filter_size // sample_time), 3), polynom_order, deriv = deriv)
        return filtered, 
    
class Blurring(ProcessNode):
    """
    Blurres timeseries by convoling it with a window if ones
    """
    outputs = ("blurred_series",)
    def _run(self, series : np.ndarray, sample_time : float, window : float) -> tuple:
        blurred = sc_signal.convolve(series, np.ones(max(int(window // sample_time), 1)), mode = 'same').astype(bool).astype(int)
        return blurred, 
    
class Edges(ProcessNode):
    """
    Detects edges in time series
    """
    outputs = ("edges_series", )

    def _run(self, series : np.ndarray, threshold : float = 0,  dir = 'both') -> tuple:
        if dir == 'rising':
            edges = self._risingEdges(series, threshold)
        elif dir == 'falling':
            edges = self._risingEdges(-series, threshold)
        else:
            edges = np.logical_or(self._risingEdges(series, threshold), self._risingEdges(-series, threshold))
        return edges, 
    
    def _risingEdges(self, y, threshold):
        """
        Find rising edges given a threshold
        """
        return np.insert((y[:-1] < threshold) & (y[1:] > threshold), 0, False)

class Detrend(ProcessNode):
    """
    Detrend times_series using scipy
    """
    outputs = ("detrended_series",)
    def _run(self, series : np.ndarray, type : str = 'linear') -> tuple:
        return sc_signal.detrend(series, type=type), 

class Correlate(ProcessNode):
    """
    Calculates correlation between series_a and series_b and and return the shift of
    series_b relative to series_a: positive shift means series_b have beed shifted 
    to the right, i.e leads series_a.
    """
    outputs = ("timeshift_maxcorr", "maxcorr", "timeshifts", "correlations")
    def _run(self, series_a : np.ndarray, series_b : np.ndarray, lower_limit : int = None, upper_limit : int = None, bipolar : bool = True) -> tuple:
        # norm factor
        # min_len = min(len(series_a), len(series_b))
        # diff = abs(len(series_a) - len(series_b))
        # norm_factor = np.concatenate((np.arange(1,min_len,1), np.full(diff + 1, min_len), np.arange(min_len-1,0,-1)))
        norm_factor = min(len(series_a), len(series_b))
        # correlate
        correlations = sc_signal.correlate((series_a - series_a.mean()) / series_a.std(), (series_b - series_b.mean()) / series_b.std()) / norm_factor
        timeshifts = np.arange(-len(series_b)+1, len(series_a), 1)
        # find bound
        mask = np.full_like(timeshifts, True, dtype = bool)
        if lower_limit is not None:
            mask[timeshifts < lower_limit] = False
        if upper_limit is not None:
            mask[timeshifts > upper_limit] = False
        # bound correlations and timeshifts
        correlations = correlations[mask]
        timeshifts = timeshifts[mask]
        # find max
        if bipolar:
            index = np.argmax(np.abs(correlations))
        else:
            index = np.argmax(correlations)

        return timeshifts[index], correlations[index], timeshifts, correlations
    
class PearsonPValue(ProcessNode):
    outputs = ("p_value",)

    def _run(self, correlation, dof) -> tuple:
        return 2*sc_stats.norm.cdf(-np.abs(np.arctan(correlation) * np.sqrt(dof - 3))), 
    
# class MaximumCorrelation(ProcessNode):
#     """
#     Calculates the maximum correlation within a set window
#     """
#     outputs = ("timeshift_maxcorr", "maxcorr", "bounded_timeshifts", "bounded_correlations")
#     def _run(self, timeshifts : np.ndarray, correlations : np.ndarray, lower_limit : int = None, upper_limit : int = None) -> tuple:
#         mask = np.full_like(timeshifts, True, dtype = bool)
#         if lower_limit is not None:
#             mask[timeshifts < lower_limit] = False
#         if upper_limit is not None:
#             mask[timeshifts > upper_limit] = False
#         correlations = correlations[mask]
#         timeshifts = timeshifts[mask]
#         index = np.argmax(np.abs(correlations))
#         return timeshifts[index], correlations[index], timeshifts, correlations

class Standardize(ProcessNode):
    """
    Standardize the timeseries
    """
    outputs = ("standardized_series",)
    def _run(self, series : np.ndarray) -> tuple:
        return (series - series.mean()) / series.std(), 

class TimeSeriesBounds(ProcessNode):
    """
    Compute lower and upper bound for time series
    """
    outputs = ("lower_bound", "upper_bound")

    def _run(self, nr_measurements : int, sample_time : float, start_time : float = 0, left_padding : float = 0, right_padding : float = 0):
        return start_time - left_padding, start_time + (nr_measurements - 1) * sample_time + right_padding

class GetTimeSeriesEvent(ProcessNode):
    outputs = ("times", "series", "event_name")
    
    def _run(self, events_df : pd.DataFrame, event_name) -> tuple:
        # inner search function
        def search(name_):
            for event_type in set(events_df["trial_type"]):
                if re.search(name_, event_type):
                    times, ts = events_df.query(f"trial_type == '{event_type}'").loc[:,["onset", "value"]].to_numpy().T
                    return times, ts, event_type
        # check if list or tuple given
        if isinstance(event_name, (list, tuple)):
            for event_n in event_name:
                r = search(event_n)
                if r is not None:
                    return r
        else:
            r = search(event_name)
            if r is not None:
                return r
        raise ValueError(f"Could not find time series event '{event_name}'.")
    
class MaskImage(ProcessNode):
    outputs = ("output_data",)
    
    def _run(self, input_img, mask_img, smoothing_fwhm = None) -> tuple:
        return masking.apply_mask(input_img, mask_img, smoothing_fwhm=smoothing_fwhm), 

class UnmaskData(ProcessNode):
    outputs = ("output_img",)
    
    def _run(self, input_data, mask_img) -> tuple:
        return masking.unmask(input_data, mask_img), 

class TemporalFilter(ProcessNode):
    outputs = ("filtered_series",)

    def _run(self, series : np.ndarray, sample_time : float, filter_freq : float, filter_order : int = 6) -> tuple:
        if isinstance(filter_freq, Iterable):
            if len(filter_freq) != 2:
                raise ValueError("'filter_freq' can either be a single value for lowpass filter of list of two values for bandpass filter.")
            btype = "bandpass"
        else:
            btype = "lowpass"

        b,a = butter(filter_order, filter_freq, btype, fs = 1 / sample_time)

        filt_func = lambda x : filtfilt(b, a, x, axis = 0, method="gust")

        if isinstance(series, pd.DataFrame):
            return series.apply(filt_func), 
        else:
            return filt_func(series), 

class TemporalFilterAll(ProcessNode):
    outputs = ("filtered_bold_data", "filtered_confounds_df", "filtered_regressor_series")

    def _run(self, sample_time : float, filter_freq : float, bold_data : np.ndarray, confounds_df : pd.DataFrame = None, regressor_series : np.ndarray = None, filter_order : int = 6) -> tuple:
        if isinstance(filter_freq, Iterable):
            if len(filter_freq) != 2:
                raise ValueError("'filter_freq' can either be a single value for lowpass filter of list of two values for bandpass filter.")
            btype = "bandpass"
        else:
            btype = "lowpass"

        b,a = butter(filter_order, filter_freq, btype, fs = 1 / sample_time)

        filt_func = lambda x : filtfilt(b, a, x, axis = 0, method="gust")

        filtered_bold_data = filt_func(bold_data)
        if confounds_df is not None:
            filtered_confounds_df = confounds_df.apply(lambda col : col if 'spike' in col.name or "constant" in col.name else filt_func(col))
        else:
            filtered_confounds_df = None
        if regressor_series is not None:
            filtered_regressor_series = filt_func(regressor_series)
        else:
            filtered_regressor_series = None

        return filtered_bold_data, filtered_confounds_df, filtered_regressor_series
    
class UpsampleAll(ProcessNode):
    """
    Upsamples a masked 4D fMRI BOLD data.
    """
    outputs = ("up_sampling_factor", "new_sample_time", "up_sampled_bold_data", "up_sampled_regressor_series")

    def _run(self, old_sample_time : float, bold_data : np.ndarray, regressor_series : np.ndarray = None,  min_sample_freq : float = 2) -> tuple:
        # up sampling factor
        up_sampling_factor = max(1, int(np.ceil(min_sample_freq * old_sample_time)))
        new_sample_time = old_sample_time / up_sampling_factor
        # intp func
        intp_func = lambda series, t_old, t_new : np.interp(t_new, t_old, series)
        # bold data
        t_old, t_new = self.oldNewTimes(bold_data.shape[0], up_sampling_factor)
        up_sampled_bold_data = np.apply_along_axis(intp_func, 0, bold_data, t_old = t_old, t_new = t_new)
        # regressor
        if regressor_series is not None:
            t_old, t_new = self.oldNewTimes(len(regressor_series), up_sampling_factor)
            up_sampled_regressor_series = intp_func(regressor_series, t_old, t_new)
        else:
            up_sampled_regressor_series = None

        return up_sampling_factor, new_sample_time, up_sampled_bold_data, up_sampled_regressor_series
    
    @staticmethod 
    def oldNewTimes(length : int, up_sampling_factor : int):
        t_old = np.arange(length)
        t_new = np.linspace(t_old[0], t_old[-1], (len(t_old) - 1) * up_sampling_factor + 1)
        return t_old, t_new
    
class Downsample(ProcessNode):
    """
    Downsamples data.
    """
    outputs = ("down_sampled_data",)

    def _run(self, down_sampling_factor : int, data : np.ndarray) -> tuple:
        return data[::down_sampling_factor], 