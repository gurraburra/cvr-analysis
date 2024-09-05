from process_control import ProcessNode
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
import scipy.signal as sc_signal
from scipy.stats import pearsonr


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
    def _run(self, timeseries_a : np.ndarray, timeseries_b : np.ndarray, time_step : float, lower_limit : int = None, upper_limit : int = None, bipolar : bool = True, window : str = None) -> tuple:
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
        timeshifts = timeshifts[mask]
        # nan values to return
        nan_values = np.nan, np.nan, timeshifts, np.full_like(timeshifts, np.nan)
        # handle leading and trailing nan values
        # series a
        timeseries_a_nan = np.isnan(timeseries_a)
        # check if all nan
        if np.all(timeseries_a_nan):
            return nan_values
        # remove trailing and leadning
        timeseries_a_leading_nan = timeseries_a_nan.argmin()
        timeseries_a_trailing_nan = timeseries_a_nan[::-1].argmin()
        # check intermediate
        if any(timeseries_a_nan[timeseries_a_leading_nan:len(timeseries_a) - timeseries_a_trailing_nan]):
            raise ValueError("Series a contains intermediate nan values.")
        timeseries_a_nan_removed = timeseries_a[timeseries_a_leading_nan:len(timeseries_a) - timeseries_a_trailing_nan]
        # series b
        timeseries_b_nan = np.isnan(timeseries_b)
        # check if all nan
        if np.all(timeseries_b_nan):
            return nan_values
        # remove leading and trailing nans
        timeseries_b_leading_nan = timeseries_b_nan.argmin()
        timeseries_b_trailing_nan = timeseries_b_nan[::-1].argmin()
        # check for intermediate nans
        if any(timeseries_b_nan[timeseries_b_leading_nan:len(timeseries_b) - timeseries_b_trailing_nan]):
            raise ValueError("Series b contains intermediate nan values.")
        timeseries_b_nan_removed = timeseries_b[timeseries_b_leading_nan:len(timeseries_b) - timeseries_b_trailing_nan]
        # check std before caculating correlations
        timeseries_a_std = timeseries_a_nan_removed.std()
        timeseries_b_std = timeseries_b_nan_removed.std()
        # if close to zero -> return directly
        if np.isclose(timeseries_a_std, 0) or np.isclose(timeseries_b_std, 0):
            return nan_values
        else:
            # length
            len_a = len(timeseries_a_nan_removed)
            len_b = len(timeseries_b_nan_removed)
            # check window
            if window is None:
                window_a = np.ones(len_a)
                window_b = np.ones(len_b)
            else:
                window_a = sc_signal.get_window(window, len_a)
                window_b = sc_signal.get_window(window, len_b)
            # applying ahmming window (basically because RapidTide suggest so)
            windowed_a = timeseries_a_nan_removed * window_a
            windowed_b = timeseries_b_nan_removed * window_b
            # correlate
            correlations = sc_signal.correlate(windowed_a, windowed_b)
            # add nan values
            correlations = np.concatenate(
                    (
                        np.full(timeseries_a_leading_nan + timeseries_b_trailing_nan, np.nan), 
                        correlations,
                        np.full(timeseries_a_trailing_nan + timeseries_b_leading_nan, np.nan)
                    )
                )
            # bound correlations and timeshifts
            correlations = correlations[mask]
            
            # find max
            if bipolar:
                index = np.nanargmax(np.abs(correlations))
            else:
                index = np.nanargmax(correlations)

            # check overlap
            idx_before_masking = np.arange(len(mask))[mask][index]
            overlap = min(idx_before_masking + 1, len_a) - max(idx_before_masking - len_b + 1, 0)
            if overlap < 2:
                return nan_values
            else:
                # correlation at idx_before_masking
                corr_at_index = pearsonr(timeseries_a_nan_removed[max(idx_before_masking - len_b + 1, 0): min(idx_before_masking + 1, len_a)], timeseries_b_nan_removed[max(len_b - idx_before_masking - 1, 0) : min(len_a + len_b - idx_before_masking - 1, len_b)]).statistic
                # correct correlation using correlation at index (otherwise windowing will lead to incorrect estimate)
                correlations *= corr_at_index / correlations[index]
                    
                return timeshifts[index], correlations[index], timeshifts, correlations
        

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