from process_control import ProcessNode
import numpy as np
from sklearn.cluster import KMeans
import scipy.signal as sc_signal
import scipy.stats as sc_stats
import pandas as pd
import re
from nilearn import masking, signal, image
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
from collections.abc import Iterable
from nilearn import glm

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
    def _run(self, series_a : np.ndarray, series_b : np.ndarray, time_step : float, lower_limit : int = None, upper_limit : int = None, bipolar : bool = True) -> tuple:
        # handle leading and trailing nan values (for shifting series)
        # series a
        series_a_nan = np.isnan(series_a)
        series_a_leading_nan = series_a_nan.argmin()
        series_a_trailing_nan = series_a_nan[::-1].argmin()
        if any(series_a_nan[series_a_leading_nan:len(series_a) - series_a_trailing_nan]):
            raise ValueError("Series a contains intermediate nan values.")
        series_a_nan_removed = series_a[series_a_leading_nan:len(series_a) - series_a_trailing_nan]
        # series b
        series_b_nan = np.isnan(series_b)
        series_b_leading_nan = series_b_nan.argmin()
        series_b_trailing_nan = series_b_nan[::-1].argmin()
        if any(series_b_nan[series_b_leading_nan:len(series_b) - series_b_trailing_nan]):
            raise ValueError("Series b contains intermediate nan values.")
        series_b_nan_removed = series_b[series_b_leading_nan:len(series_b) - series_b_trailing_nan]
        # norm factor
        # min_len = min(len(series_a), len(series_b))
        # diff = abs(len(series_a) - len(series_b))
        # norm_factor = np.concatenate((np.arange(1,min_len,1), np.full(diff + 1, min_len), np.arange(min_len-1,0,-1)))
        norm_factor = min(len(series_a), len(series_b))
        # correlate
        correlations = sc_signal.correlate(
                        (series_a_nan_removed - series_a_nan_removed.mean()) / series_a_nan_removed.std(), 
                            (series_b_nan_removed - series_b_nan_removed.mean()) / series_b_nan_removed.std()) / norm_factor
        # add nan values
        correlations = np.concatenate(
                (
                    np.full(series_a_leading_nan + series_b_trailing_nan, np.nan), 
                    correlations,
                    np.full(series_a_trailing_nan + series_b_leading_nan, np.nan)
                )
            )
        # timeshifts
        timeshifts = (np.arange(-len(series_b)+1, len(series_a), 1)) * time_step
        # find bound
        mask = np.full_like(timeshifts, True, dtype = bool)
        if lower_limit is not None:
            mask[timeshifts < lower_limit] = False
        if upper_limit is not None:
            mask[timeshifts > upper_limit] = False
        # find max
        if any(mask):
            # bound correlations and timeshifts
            correlations = correlations[mask]
            timeshifts = timeshifts[mask]
            if bipolar:
                index = np.nanargmax(np.abs(correlations))
            else:
                index = np.nanargmax(correlations)
        elif lower_limit != None and lower_limit == upper_limit:
            # pick value closet to the limits
            index = np.argmin(np.abs(timeshifts - lower_limit))
        else:
            raise ValueError("Incorrect limits specified.")

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
    
    def _run(self, input_img, mask_img, smoothing_fwhm = None, ensure_no_mixing = True) -> tuple:
        # if voxels should not be mixed in by smoothing -> mask and unmask before smoothing
        if ensure_no_mixing:
            input_img = image.math_img('img * np.array(mask, dtype=bool)[...,None]', img = input_img, mask = mask_img)
        return masking.apply_mask(input_img, mask_img, smoothing_fwhm=smoothing_fwhm), 

class UnmaskData(ProcessNode):
    outputs = ("output_img",)
    
    def _run(self, input_data, mask_img) -> tuple:
        return masking.unmask(input_data, mask_img), 

class TemporalFilter(ProcessNode):
    outputs = ("filtered_series",)

    def _run(self, series : np.ndarray, sample_time : float, filter_freq : float, filter_order : int = 6) -> tuple:
        # check if filter freq is none, return directly
        if filter_freq is None:
            return series
        # otherwise filter data
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
        # check if filter freq is none, return directly
        if filter_freq is None:
            return bold_data, confounds_df, regressor_series
        # otherwise filter data
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
    outputs = ("up_sampling_factor", "new_sample_time", "up_sampled_bold_data", "up_sampled_confounds_df", "up_sampled_regressor_series")

    def _run(self, old_sample_time : float, bold_data : np.ndarray, confounds_df : pd.DataFrame, regressor_series : np.ndarray = None,  min_sample_freq : float = 2) -> tuple:
        # up sampling factor
        up_sampling_factor = max(1, int(np.ceil(min_sample_freq * old_sample_time)))
        new_sample_time = old_sample_time / up_sampling_factor
        # intp func
        intp_func = lambda series, t_old, t_new : np.interp(t_new, t_old, series)
        # bold data
        t_old, t_new = self.oldNewTimes(bold_data.shape[0], up_sampling_factor)
        up_sampled_bold_data = np.apply_along_axis(intp_func, 0, bold_data, t_old = t_old, t_new = t_new)
        # confounds
        up_sampled_confounds_df = confounds_df.apply(intp_func, t_old = t_old, t_new = t_new)
        # regressor
        if regressor_series is not None:
            t_old, t_new = self.oldNewTimes(len(regressor_series), up_sampling_factor)
            up_sampled_regressor_series = intp_func(regressor_series, t_old, t_new)
        else:
            up_sampled_regressor_series = None

        return up_sampling_factor, new_sample_time, up_sampled_bold_data, up_sampled_confounds_df, up_sampled_regressor_series
    
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

class AlignTimeSeries(ProcessNode):
    outputs = ("aligned_series",)

    def _run(self, series : np.ndarray, timeshift : float, time_step : float, length : int, fill_nan : bool = True) -> tuple:
        # convert to index
        timeshift_index = int(round(timeshift / time_step, 0))
        # fill nan
        if fill_nan:
            fill_l = np.nan
            fill_r = np.nan
        else:
            fill_l = series[0]
            fill_r = series[-1]
        # handled timeshift
        if timeshift_index <= 0:
            series = series[-timeshift_index:]
        else:
            series = np.concatenate((np.full(timeshift_index, fill_l), series))
        # make sure equal length
        diff = length - len(series)
        if diff >= 0:
            series = np.concatenate((series, np.full(diff, fill_r)))
        else:
            series = series[:diff]
        
        return series, 


class CleanData(ProcessNode):
    outputs = ("cleaned_data", )

    def _run(self, data : np.ndarray, sample_time : float, detrend : bool = False, filter_freq : float = None, filter_order : int = 6, confounds_df : pd.DataFrame = None, standardize : object = False) -> tuple:
        # check filter freq
        if isinstance(filter_freq, Iterable):
            if len(filter_freq) != 2:
                raise ValueError("'filter_freq' can either be a single value for lowpass filter of list of two values for bandpass filter.")
            high_pass = filter_freq[0]
            low_pass = filter_freq[1]
        else:
            high_pass = None
            low_pass = filter_freq
        # check data type
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                cleaned_data = signal.clean(data[:,None], detrend=detrend, t_r=sample_time, low_pass=low_pass, high_pass=high_pass, butterworth_order = filter_order, confounds=confounds_df, standardize=standardize)[:,0]
            elif data.ndim == 2:
                cleaned_data = signal.clean(data, detrend=detrend, t_r=sample_time, low_pass=low_pass, high_pass=high_pass, butterworth_order = filter_order, confounds=confounds_df, standardize=standardize)
            else:
                ValueError(f"Incompatible number of data dimensions: {data.ndim}.")
        elif isinstance(data, pd.DataFrame):
            cleaned_data = pd.DataFrame(signal.clean(data.to_numpy(), detrend=detrend, t_r=sample_time, low_pass=low_pass, high_pass=high_pass, butterworth_order = filter_order, confounds=confounds_df, standardize=standardize),
                                        index=data.index, columns=data.columns)
        else:
            raise ValueError(f"Incompatible data type: {type(data)}.")
        
        return cleaned_data, 

class DetrendAll(ProcessNode):
    """
    Upsamples a masked 4D fMRI BOLD data.
    """
    outputs = ("detrended_bold_data", "detrended_confounds_df", "detrended_regressor_series")

    def _run(self, bold_data : np.ndarray, confounds_df : pd.DataFrame, regressor_series : np.ndarray = None, detrend_order : int = 1) -> tuple:
        # bold + confounds 
        bids_confounds = glm.first_level.make_first_level_design_matrix(np.arange(bold_data.shape[0]), drift_model = "polynomial", drift_order = detrend_order).drop(columns="constant")
        detrend_bold_data = signal.clean(bold_data, confounds=bids_confounds, detrend=False, standardize=False)
        detrend_confounds_df = pd.DataFrame(signal.clean(confounds_df.to_numpy(), confounds=bids_confounds, detrend=False, standardize=False), 
                                                index = confounds_df.index, columns = confounds_df.columns)
        # regressor
        if regressor_series is not None:
            regressor_confounds = glm.first_level.make_first_level_design_matrix(np.arange(regressor_series.shape[0]), drift_model = "polynomial", drift_order = detrend_order).drop(columns="constant")
            detrend_regressor_series = signal.clean(regressor_series[:,None], confounds=regressor_confounds, detrend=False, standardize=False)[:,0]
        else:
            detrend_regressor_series = None

        return detrend_bold_data, detrend_confounds_df, detrend_regressor_series