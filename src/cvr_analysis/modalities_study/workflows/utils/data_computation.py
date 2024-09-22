from process_control import ProcessNode
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
import scipy.signal as sc_signal
from scipy.stats import pearsonr
from scipy.fft import rfft, irfft
from sklearn.feature_selection import mutual_info_regression
from nilearn import maskers
from scipy.ndimage import median_filter, uniform_filter
from nilearn import image
from sklearn.decomposition import PCA

class BaselinePlateau(ProcessNode):
    """
    Finds baseline and plateoa by using a kmeans algorithm
    """
    outputs = ("baseline", "plateau")

    def _run(self, timeseries : np.ndarray) -> tuple:
        kmeans_centers = KMeans(n_clusters = 2, n_init="auto").fit(timeseries.reshape(-1, 1)).cluster_centers_
        return kmeans_centers.min(), kmeans_centers.max()
    
class KmeanNormTimeSeries(ProcessNode):
    """
    Norm timeseries
    """
    outputs = ("normed_timeseries", "normed_baseline", "normed_plateau")

    def _run(self, timeseries : np.ndarray) -> tuple:
        kmeans_centers = KMeans(n_clusters = 2, n_init="auto").fit(timeseries.reshape(-1, 1)).cluster_centers_
        baseline, plateau = kmeans_centers.min(), kmeans_centers.max()
    
        return timeseries / baseline * 100, 100, plateau / baseline * 100
    

class BaselineTimeSeries(ProcessNode):
    """
    Calculate baseline for timeseries
    """
    outputs = ("baseline", )

    def _run(self, timeseries : np.ndarray, time_step : float, baseline_strategy : str = "overall-mean") -> tuple:
        # overall mean
        if baseline_strategy == 'overall-mean':
            baseline = timeseries.mean(axis=0)
        # initial mean
        elif baseline_strategy.startswith('initial-mean-'):
            try:
                mean_time = float(baseline_strategy[len('initial-mean-'):])
            except:
                raise ValueError(f"'baseline-stategy' needs to either be 'overall-mean', 'initial-mean-xx', 'final-mean-xx' or 'endpoints-interpolate-xx', where 'xx' is the mean-time in seconds")
            nr_idx = max(1, int(mean_time / time_step))
            baseline = timeseries[:nr_idx].mean(axis=0)
        # final mean
        elif baseline_strategy.startswith('final-mean-'):
            try:
                mean_time = float(baseline_strategy[len('final-mean-'):])
            except:
                raise ValueError(f"'baseline-stategy' needs to either be 'overall-mean', 'initial-mean-xx', 'final-mean-xx' or 'endpoints-interpolate-xx', where 'xx' is the mean-time in seconds")
            nr_idx = max(1, int(mean_time / time_step))
            baseline = timeseries[-nr_idx:].mean(axis=0)
        # endpoints interpolate
        elif baseline_strategy.startswith('endpoints-interpolate-'):
            try:
                mean_time = float(baseline_strategy[len('endpoints-interpolate-'):])
            except:
                raise ValueError(f"'baseline-stategy' needs to either be 'overall-mean', 'initial-mean-xx', 'final-mean-xx' or 'endpoints-interpolate-xx', where 'xx' is the mean-time in seconds")
            nr_idx = max(1, int(mean_time / time_step))
            # check timeseries type
            if isinstance(timeseries, np.ndarray) and (timeseries.ndim == 1 or timeseries.ndim == 2):
                if timeseries.ndim == 2:
                    baseline = self._endPointsInterpolate(timeseries, nr_idx)
                else:
                    baseline = self._endPointsInterpolate(timeseries[:,None], nr_idx)[:,0]
            elif isinstance(timeseries, pd.DataFrame):
                baseline = pd.DataFrame(self._endPointsInterpolate(timeseries.to_numpy(), nr_idx), 
                                                    index = timeseries.index, columns = timeseries.columns)
            else:
                raise ValueError(f"timeseries must be 1D/2D numpy array or pandas dataframe, '{type(timeseries)}' was given")
        else:
            raise ValueError(f"'baseline-stategy' needs to either be 'overall-mean', 'initial-mean-xx', 'final-mean-xx' or 'endpoints-interpolate-xx', where 'xx' is the mean-time in seconds")

        return baseline, 

    def _endPointsInterpolate(self, timeseries, nr_idx):
        # get length
        l = timeseries.shape[0]
        # get initial and final mean
        initial_mean, final_mean = timeseries[:nr_idx].mean(axis = 0), timeseries[-nr_idx:].mean(axis = 0)
        # get diff and sum
        end_point_diff = final_mean - initial_mean
        end_point_sum = final_mean + initial_mean
        # create linear trend
        end_points_interpolate = (np.arange(l)[:,None] - (l - 1) / 2) * end_point_diff[None,:] / (l - nr_idx) + end_point_sum / 2
        return end_points_interpolate
    
class PercentageChangeTimeSeries(ProcessNode):
    """
    Percentage change timeseries
    """

    outputs = ("percentage_timeseries", "baseline_timeseries")

    def _run(self, timeseries : np.ndarray, baseline : np.ndarray,) -> tuple:
        # return if none
        if timeseries is None:
            return None, None
        # mask
        mask = np.logical_or(np.isclose(baseline,0), baseline < 0)
        baseline_with_nan = np.where(mask, np.nan, baseline)
        # check dim of baseline
        if baseline.ndim == 2:
            cols_with_nan = np.any(np.isnan(baseline_with_nan), axis = 0)
            baseline_with_nan[:,cols_with_nan] = np.nan
        
        # percentage change
        percentage_timeseries = (timeseries - baseline) / baseline_with_nan * 100
    
        return percentage_timeseries, timeseries - baseline
    
class StandardizeTimeSeries(ProcessNode):
    """
    Standardize timeseries
    """
    outputs = ("standardized_timeseries", )

    def _run(self, timeseries : np.ndarray) -> tuple:
        # return if none
        if timeseries is None:
            return None
        # check stategy
        return (timeseries - timeseries.mean(axis = 0)) / timeseries.std(axis = 0), 

class RMSTimeSeries(ProcessNode):
    """
    Power of timeseries
    """
    outputs = ("rms", )

    def _run(self, timeseries : np.ndarray) -> tuple:
        rms = np.sqrt(np.sum(np.square(timeseries), axis = 0) / timeseries.shape[0])
    
        return rms, 
    
class Correlate(ProcessNode):
    """
    Calculates correlation between timeseries_a and timeseries_b and and return the shift of
    timeseries_b relative to timeseries_a: positive shift means timeseries_b have beed shifted 
    to the right, i.e leads timeseries_a.
    """
    outputs = ("timeshift_maxcorr", "maxcorr", "timeshifts", "correlations")
    def _run(self, timeseries_a : np.ndarray, timeseries_b : np.ndarray, time_step : float, lower_limit : int = None, upper_limit : int = None, bipolar : bool = True, window : str = None, phat : bool = False, multi_peak_strategy : str = None, ref_timeshift : float = 0) -> tuple:
        # timeshifts
        timeshifts = (np.arange(-len(timeseries_b)+1, len(timeseries_a), 1)) * time_step
        # mask
        mask = np.full_like(timeshifts, True, dtype = bool)
        if lower_limit is not None:
            mask[timeshifts < lower_limit] = False
        if upper_limit is not None:
            mask[timeshifts > upper_limit] = False
        # check mask
        if not np.any(mask):
            if lower_limit is not None and lower_limit == upper_limit:
                # pick value closet to the limits
                index = np.argmin(np.abs(timeshifts - lower_limit))
                mask[index] = True
            else:
                raise ValueError("Incorrect limits specified.")
        # mask timeshifts
        timeshifts_masked = timeshifts[mask]
        # nan values to return
        nan_values = np.nan, np.nan, timeshifts_masked, np.full_like(timeshifts_masked, np.nan)
        # handle leading and trailing nan values
        # series a
        timeseries_a_nan_removed, timeseries_a_leading_nan, timeseries_a_trailing_nan = self.removeNan(timeseries_a)
        # check if all nan
        if timeseries_a_nan_removed.size == 0:
            return nan_values
        # series b
        timeseries_b_nan_removed, timeseries_b_leading_nan, timeseries_b_trailing_nan = self.removeNan(timeseries_b)
        # check if all nan
        if timeseries_b_nan_removed.size == 0:
            return nan_values
        # check std before caculating correlations
        timeseries_a_std = timeseries_a_nan_removed.std()
        timeseries_b_std = timeseries_b_nan_removed.std()
        # if close to zero -> return directly
        if np.isclose(timeseries_a_std, 0) or np.isclose(timeseries_b_std, 0):
            return nan_values
        else:
            # length
            len_a_nan_removed = len(timeseries_a_nan_removed)
            len_b_nan_removed = len(timeseries_b_nan_removed)
            # check window
            if window is None:
                window_a = np.ones(len_a_nan_removed)
                window_b = np.ones(len_b_nan_removed)
            else:
                window_a = sc_signal.get_window(window, len_a_nan_removed)
                window_b = sc_signal.get_window(window, len_b_nan_removed)
            # applying ahmming window (basically because RapidTide suggest so)
            windowed_a = timeseries_a_nan_removed * window_a
            windowed_b = timeseries_b_nan_removed * window_b
            # correlate
            correlations_nan_removed = self.xcorr(windowed_a, windowed_b, phat)#sc_signal.correlate(windowed_a, windowed_b)
            # add nan values
            correlations = np.concatenate(
                    (
                        np.full(timeseries_a_leading_nan + timeseries_b_trailing_nan, np.nan), 
                        correlations_nan_removed,
                        np.full(timeseries_a_trailing_nan + timeseries_b_leading_nan, np.nan)
                    )
                )
            # bound correlations
            correlations_masked = correlations[mask]
            # masked indices to convert between index and masked index
            masked_indices = np.arange(len(mask))[mask]
            
            # peak finding values (before adding nan)
            if bipolar:
                p_corr_masked = np.abs(correlations_masked)
            else:
                p_corr_masked = correlations_masked
            # check peak strategy 
            if multi_peak_strategy is None:
                index_masked = np.nanargmax(p_corr_masked)
            elif multi_peak_strategy in ["max", "ref", "mi"]: 
                # peak finding algorithm
                p_corr_masked_removed_nan, p_corr_masked_leading_nan, _ = self.removeNan(p_corr_masked)
                peaks_idx_masked = sc_signal.find_peaks(p_corr_masked_removed_nan, height = 0)[0] + p_corr_masked_leading_nan
                
                if peaks_idx_masked.size == 0:
                    index_masked = np.nanargmax(p_corr_masked)
                elif peaks_idx_masked.size == 1:
                    index_masked = peaks_idx_masked[0]
                else:
                    if multi_peak_strategy == 'max':
                        # choose peak with largest value
                        index_masked = peaks_idx_masked[p_corr_masked[peaks_idx_masked].argmax()]
                    elif multi_peak_strategy == 'ref':
                        # choose peak closes to reference timeshift
                        index_masked = peaks_idx_masked[np.abs(timeshifts_masked[peaks_idx_masked] - ref_timeshift).argmin()]
                    elif multi_peak_strategy == 'mi':
                        # choose the one with largest mutual information
                        mi = []
                        for peak_idx_masked in peaks_idx_masked:
                            # convert masked peak to candidate peak
                            peak_idx = masked_indices[peak_idx_masked]
                            # convert to nan peak
                            peak_idx_nan = peak_idx - timeseries_a_leading_nan - timeseries_b_trailing_nan
                            # shift timeseries
                            ser_a_mi, ser_b_mi = self.timeshiftedSeries(timeseries_a_nan_removed, timeseries_b_nan_removed, peak_idx_nan)
                            # calculate mi
                            mi.append(mutual_info_regression(ser_b_mi.reshape(-1,1), ser_a_mi))
                        index_masked = peaks_idx_masked[np.argmax(mi)]
            else:
                raise ValueError(f"'multi_peak_strategy' can either be None, 'max', 'ref' or 'mi', '{multi_peak_strategy}' was given")
                    
            # index before applying mask
            index = np.arange(len(mask))[mask][index_masked]
            # index before adding nan
            index_nan = index - timeseries_a_leading_nan - timeseries_b_trailing_nan
            # timeshifted nan series
            timeshifted_a_nan_removed, timeshifted_b_nan_removed = self.timeshiftedSeries(timeseries_a_nan_removed, timeseries_b_nan_removed, index_nan)
            # check overlap
            if timeshifted_a_nan_removed.size < 2:
                return nan_values
            else:
                # correlation at idx_before_masking
                corr_at_index = pearsonr(timeshifted_a_nan_removed, timeshifted_b_nan_removed).statistic
                # correct correlation using correlation at index (otherwise windowing will lead to incorrect estimate)
                correlations_masked *= corr_at_index / correlations_masked[index_masked]
                    
                return timeshifts_masked[index_masked], correlations_masked[index_masked], timeshifts_masked, correlations_masked
            
    def removeNan(self, timeseries):
        # series is nan
        timeseries_nan = np.isnan(timeseries)
        # remove trailing and leadning
        # check if all nan
        if np.all(timeseries_nan):
            timeseries_leading_nan = len(timeseries)
        else:
            timeseries_leading_nan = timeseries_nan.argmin()
        timeseries_trailing_nan = timeseries_nan[::-1].argmin()
        # check intermediate
        if np.any(timeseries_nan[timeseries_leading_nan:len(timeseries) - timeseries_trailing_nan]):
            raise ValueError("Series a contains intermediate nan values.")
    
        return timeseries[timeseries_leading_nan:len(timeseries) - timeseries_trailing_nan], timeseries_leading_nan, timeseries_trailing_nan
        
    def xcorr(self, s1, s2, phat = False):
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        length = len(s1) + len(s2) - 1
        f_s1 = rfft(s1, length)
        f_s2 = rfft(s2[::-1].conj(), length)

        f_s = f_s1 * f_s2
        if phat:
            denom = abs(f_s)
            denom[denom < 1e-6] = 1e-6
            f_s = f_s / denom  # This line is the only difference between GCC-PHAT and normal cross correlation
            # the phat seems to be sensitive to numerical issues at the endpoints leading to large values, therefore set these values to zero
            corr = irfft(f_s, length)
            zero_crossing = np.sign(corr[1:-2] * corr[2:-1]) < 0
            leading = zero_crossing.argmax()
            trailing = zero_crossing[::-1].argmax()
            lim = len(corr) / 20
            if leading < lim:
                corr[ : 3 * (leading + 1)] = 0
            if trailing < lim:
                corr[len(corr) - 3 * (trailing + 1) : ] = 0
            return corr
        else:
            return irfft(f_s, length)
        
    def timeshiftedSeries(self, ser_a, ser_b, timeshift_idx):
        len_a, len_b = len(ser_a), len(ser_b)
        slice_a = slice(max(timeshift_idx - len_b + 1, 0), min(max(timeshift_idx + 1, 0), len_a))
        slice_b = slice(max(len_b - timeshift_idx - 1, 0), min(max(len_a + len_b - timeshift_idx - 1, 0), len_b))
        
        return ser_a[slice_a], ser_b[slice_b]
    
class HistPeak(ProcessNode):
    outputs = ("histogram_peak", )

    def _run(self, values : np.ndarray) -> tuple:
        # count occurences
        unique, counts = np.unique(values, return_counts=True)
        return unique[counts.argmax()], 

class FilterTimeshifts(ProcessNode):
    outputs = ("filtered_timeshifts",)

    def _run(self, timeseries_masker : maskers.NiftiMasker, timeshifts : np.ndarray, maxcorrs : float, maxcorr_threshold : float = 0.5, size : int = 3, filter_type : str = 'median') -> tuple:
        # check if None
        if filter_type is None:
            return timeshifts
        # copy data
        new_timeshifts = timeshifts.copy()
        # convert to img
        new_timeshifts_img = timeseries_masker.inverse_transform(new_timeshifts)
        # check filter type
        if filter_type == 'median':
            filter = median_filter
        elif filter_type == 'mean':
            filter = uniform_filter
        else:
            raise ValueError(f"'filter_type' can either be 'median' or 'mean', '{filter_type}' was given")
        # filter data
        filtered_timeshift_3d_data = filter(new_timeshifts_img.get_fdata(), size = size)
        # convert back to array
        filtered_timesshift_data = timeseries_masker.transform(
            image.new_img_like(new_timeshifts_img, filtered_timeshift_3d_data)
        )[0]
        # get values to mask
        mask = np.abs(maxcorrs) < maxcorr_threshold
        # update those values
        new_timeshifts[mask] = filtered_timesshift_data[mask]
        return new_timeshifts,
    
class PCAReducedTimeSeries(ProcessNode):
    outputs = ("reduced_timeseries", "pca_components", "explained_variance_ratio")

    def _run(self, timeseries : np.ndarray, explained_variance : float = 0.5) -> tuple:
        # assert explained variance in between 0 and 1
        assert explained_variance > 0 and explained_variance < 1, f"'explained_variance' needs to be between 0 and 1"
        # create PCA solver
        pca = PCA(n_components=explained_variance)
        # apply pca
        reduced_timeseries = pca.inverse_transform(pca.fit_transform(timeseries.T)).T

        return reduced_timeseries, pca.components_, pca.explained_variance_ratio_

class AlignTimeSeries(ProcessNode):
    outputs = ("aligned_timeseries",)

    def _run(self, timeseries : np.ndarray, timeshift : float, time_step : float, length : int, fill_nan : bool = True) -> tuple:
        # check if nan
        if np.isnan(timeshift):
            return np.full(length, np.nan)
        # convert to index
        timeshift_index = int(round(timeshift / time_step, 0))
        # fill nan
        if fill_nan:
            fill_l = np.nan
            fill_r = np.nan
        else:
            fill_l = timeseries[0]
            fill_r = timeseries[-1]
        # handle timeshift
        if timeshift_index <= 0:
            timeseries = timeseries[-timeshift_index:]
        else:
            timeseries = np.concatenate((np.full(timeshift_index, fill_l), timeseries))
        # make sure equal length
        diff = length - len(timeseries)
        if diff >= 0:
            timeseries = np.concatenate((timeseries, np.full(diff, fill_r)))
        else:
            timeseries = timeseries[:diff]
        
        return timeseries, 

class RegressCVR(ProcessNode):
    """
    Performs linear regression between regressir data och bold data.
    """
    outputs = ("dof", "nr_predictors", "regressor_beta", "design_matrix", "betas", "predictions", "r_squared",  "adjusted_r_squared", "tsnr")

    def _run(self, bold_ts : np.ndarray, regressor_timeseries : np.ndarray, confounds_df : pd.DataFrame = None, confound_regressor_correlation_threshold : float = None) -> tuple:
        # check confounds_df
        if confounds_df is None:
            confounds_df = pd.DataFrame([], index = range(len(bold_ts)))
        # create design matrix
        design_matrix = pd.concat((pd.Series(regressor_timeseries, name = "regressor"), confounds_df.reset_index(drop=True)), axis = 1)
        # add constant
        if "constant" not in design_matrix:
            design_matrix["constant"] = 1
        # find nan rows
        nan_entries = design_matrix.isna().any(axis=1) | np.isnan(bold_ts)
        # check if no valid entries
        if np.all(nan_entries):
            # return all nans
            return 0, design_matrix.shape[1], np.nan, design_matrix, np.full(design_matrix.shape[1], np.nan), np.full_like(bold_ts, np.nan), np.nan, np.nan, np.nan
        non_nan_dm = design_matrix[~nan_entries]
        non_nan_bs = bold_ts[~nan_entries]
        # threshold conofunds
        if confound_regressor_correlation_threshold is not None and not confounds_df.empty:
            # get standardized confounds
            confounds = non_nan_dm[confounds_df.columns].to_numpy()
            confounds = (confounds - confounds.mean(axis = 0)) / confounds.std(axis = 0)
            # get standardized regressor
            regressor = non_nan_dm["regressor"].to_numpy()
            regressor = (regressor - regressor.mean(axis = 0)) / regressor.std(axis = 0)
            # calculate correlation between regressor and confounds
            confound_corr = confounds.T @ regressor / len(regressor) 
            # threshold
            confounds_above_thr = confounds_df.columns[np.abs(confound_corr) > confound_regressor_correlation_threshold]
            # drop
            design_matrix = design_matrix.drop(columns = confounds_above_thr)
            non_nan_dm = non_nan_dm.drop(columns = confounds_above_thr)
        # get dimensions
        n, p = non_nan_dm.shape
        # regress
        reg = lm.LinearRegression(fit_intercept=False).fit(non_nan_dm, non_nan_bs)
        betas = reg.coef_
        non_nan_pred = reg.predict(non_nan_dm)
        # r_squared
        r_squared = r2_score(non_nan_bs, non_nan_pred)
        # adjusted r-squared
        if n - p - 1 > 0:
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        else:
            adjusted_r_squared = 0
        # tsnr
        residual_sum_of_squares = np.sum((non_nan_bs - non_nan_pred)**2)
        # the baseline mean signal is 100 since we assume the bold signal has been normed by the basline and multiplied by 100
        tsnr = 100 / np.sqrt(residual_sum_of_squares / n) if residual_sum_of_squares > 0 else np.nan
        # regressor beta
        regressor_beta = betas[design_matrix.columns.get_loc("regressor")]
        # return
        return n, p, regressor_beta, design_matrix, betas, design_matrix @ betas, r_squared, adjusted_r_squared, tsnr