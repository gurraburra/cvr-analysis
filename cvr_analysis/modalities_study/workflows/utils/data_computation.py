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
    
class NormTimeSeries(ProcessNode):
    """
    Norm timeseries
    """
    outputs = ("normed_timeseries", "normed_baseline", "normed_plateau")

    def _run(self, timeseries : np.ndarray) -> tuple:
        kmeans_centers = KMeans(n_clusters = 2, n_init="auto").fit(timeseries.reshape(-1, 1)).cluster_centers_
        baseline, plateau = kmeans_centers.min(), kmeans_centers.max()
    
        return timeseries / baseline * 100, 100, plateau / baseline * 100
    
class Correlate(ProcessNode):
    """
    Calculates correlation between timeseries_a and timeseries_b and and return the shift of
    timeseries_b relative to timeseries_a: positive shift means timeseries_b have beed shifted 
    to the right, i.e leads timeseries_a.
    """
    outputs = ("timeshift_maxcorr", "maxcorr", "timeshifts", "correlations")
    def _run(self, timeseries_a : np.ndarray, timeseries_b : np.ndarray, time_step : float, lower_limit : int = None, upper_limit : int = None, bipolar : bool = True, window : str = None) -> tuple:
        # handle leading and trailing nan values
        # series a
        timeseries_a_nan = np.isnan(timeseries_a)
        timeseries_a_leading_nan = timeseries_a_nan.argmin()
        timeseries_a_trailing_nan = timeseries_a_nan[::-1].argmin()
        if any(timeseries_a_nan[timeseries_a_leading_nan:len(timeseries_a) - timeseries_a_trailing_nan]):
            raise ValueError("Series a contains intermediate nan values.")
        timeseries_a_nan_removed = timeseries_a[timeseries_a_leading_nan:len(timeseries_a) - timeseries_a_trailing_nan]
        # series b
        timeseries_b_nan = np.isnan(timeseries_b)
        timeseries_b_leading_nan = timeseries_b_nan.argmin()
        timeseries_b_trailing_nan = timeseries_b_nan[::-1].argmin()
        if any(timeseries_b_nan[timeseries_b_leading_nan:len(timeseries_b) - timeseries_b_trailing_nan]):
            raise ValueError("Series b contains intermediate nan values.")
        timeseries_b_nan_removed = timeseries_b[timeseries_b_leading_nan:len(timeseries_b) - timeseries_b_trailing_nan]

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

        # check std before caculating correlations
        timeseries_a_std = timeseries_a_nan_removed.std()
        timeseries_b_std = timeseries_b_nan_removed.std()
        # standardize function
        def stand(ser):
            return (ser-ser.mean()) / ser.std()
        # if close to zero -> return directly
        if np.isclose(timeseries_a_std, 0) or np.isclose(timeseries_b_std, 0):
            return 0, 0, timeshifts, np.zeros_like(timeshifts)
        else:
            # standardized timeseries
            stand_a = stand(timeseries_a_nan_removed)
            stand_b = stand(timeseries_b_nan_removed)
            len_a = len(stand_a)
            len_b = len(stand_b)
            # check window
            if window is None:
                window_a = np.ones(len_a)
                window_b = np.ones(len_b)
            else:
                window_a = sc_signal.get_window(window, len_a)
                window_b = sc_signal.get_window(window, len_b)
            # applying ahmming window (basically because RapidTide suggest so)
            windowed_stand_a = stand_a * window_a
            windowed_stand_b = stand_b * window_b
            # correlate
            correlations = sc_signal.correlate(windowed_stand_a, windowed_stand_b)
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
                return 0, 0, timeshifts, np.zeros_like(timeshifts)
            else:
                # correlation at idx_before_masking
                corr_at_index = pearsonr(stand_a[max(idx_before_masking - len_b + 1, 0): min(idx_before_masking + 1, len_a)], stand_b[max(len_b - idx_before_masking - 1, 0) : min(len_a + len_b - idx_before_masking - 1, len_b)]).statistic
                # correct correlation using correlation at index (otherwise windowing will lead to incorrect estimate)
                correlations *= np.abs(corr_at_index / correlations[index])
                    
                return timeshifts[index], correlations[index], timeshifts, correlations
        

class AlignTimeSeries(ProcessNode):
    outputs = ("aligned_timeseries",)

    def _run(self, timeseries : np.ndarray, timeshift : float, time_step : float, length : int, fill_nan : bool = True) -> tuple:
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
    outputs = ("dof", "nr_predictors", "design_matrix", "betas", "predictions", "r_squared",  "adjusted_r_squared", "tsnr")

    def _run(self, bold_ts : np.ndarray, regressor_timeseries : np.ndarray = None, confounds_df : pd.DataFrame = None) -> tuple:
        # check confounds_df
        if confounds_df is None:
            confounds_df = pd.DataFrame([], index = range(len(bold_ts)))
        # cceate design matrix
        if regressor_timeseries is not None:
            # combine
            design_matrix = pd.concat((pd.Series(regressor_timeseries, name = "regressor"), confounds_df.reset_index(drop=True)), axis = 1)
        else:
            design_matrix = confounds_df.reset_index(drop=True)
        # add constant
        if "constant" not in design_matrix:
            design_matrix["constant"] = 1
        # find nan rows
        nan_entries = design_matrix.isna().any(axis=1) | np.isnan(bold_ts)
        non_nan_dm = design_matrix[~nan_entries]
        non_nan_bs = bold_ts[~nan_entries]
        # get dimensions
        n, p = non_nan_dm.shape
        # regress
        reg = lm.LinearRegression(fit_intercept=False).fit(non_nan_dm, non_nan_bs)
        betas = reg.coef_
        non_nan_pred = reg.predict(non_nan_dm)
        # r_squared
        r_squared = r2_score(non_nan_bs, non_nan_pred)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        # tsnr
        residual_sum_of_squares = np.sum((non_nan_bs - non_nan_pred)**2)
        mean_signal = np.mean(non_nan_bs)
        tsnr = mean_signal / np.sqrt(residual_sum_of_squares / n) if residual_sum_of_squares > 0 else np.nan
        # return
        return n, p, design_matrix, betas, design_matrix @ betas, r_squared, adjusted_r_squared, tsnr
    
class CVRAmplitude(ProcessNode):
    outputs = ("cvr_amplitude", "regressor_beta")

    def _run(self, design_matrix : pd.DataFrame, betas : np.ndarray, regressor_baseline : float = None) -> tuple:
        mean_features = design_matrix.mean()
        if regressor_baseline is not None:
            mean_features["regressor"] == regressor_baseline
        reg_idx = mean_features.index.get_loc("regressor")
        mean_predict = mean_features @ betas
        if np.isclose(mean_predict, 0):
            return 0.0, betas[reg_idx]
        else:
            return betas[reg_idx] / mean_predict * 100, betas[reg_idx]
    