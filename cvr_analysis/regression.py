from process_control import *

from bids import BIDSLayout
from nilearn import image, masking
import pandas as pd
import numpy as np
from scipy import interpolate
import scipy.signal as sc_signal
import scipy.stats as sc_stats
import sklearn.linear_model as lm
from sklearn.cluster import KMeans
import multiprocess as mp
from pathlib import Path
import os
import statsmodels.api as sm



class RegressCVR(ProcessNode):
    """
    Performs linear regression between regressir data och bold data.
    """
    outputs = ("dof", "design_matrix", "betas", "r_squared",  "r_squared_adjusted", "predictions")

    def _run(self, bold_series : np.ndarray, regressor_series : np.ndarray = None, confounds_df : pd.DataFrame = None) -> tuple:
        # check confounds_df
        if confounds_df is None:
            confounds_df = pd.DataFrame([], index = range(len(bold_series)))
        # cceate design matrix
        if regressor_series is not None:
            # combine
            design_matrix = pd.concat((pd.Series(regressor_series, name = "regressor"), confounds_df.reset_index(drop=True)), axis = 1)
        else:
            design_matrix = confounds_df.reset_index(drop=True)
        # add constant
        if "constant" not in design_matrix:
            design_matrix["constant"] = 1
        # find nan rows
        nan_rows = design_matrix.isna().any(axis=1)
        non_nan_dm = design_matrix[~nan_rows]
        non_nan_bs = bold_series[~nan_rows]
        n, p = non_nan_dm.shape
        # regress
        reg = lm.LinearRegression(fit_intercept=False).fit(non_nan_dm, non_nan_bs)
        betas = reg.coef_
        non_nan_pred = reg.predict(non_nan_dm)
        # r_squared
        residuals = non_nan_bs - non_nan_pred
        residual_sum_of_squares = residuals.T @ residuals
        residual_constant_model = non_nan_bs - np.mean(non_nan_bs)
        residual_sum_of_squares_constant_model = residual_constant_model.T @ residual_constant_model
        r_squared = 1 - residual_sum_of_squares / residual_sum_of_squares_constant_model if residual_sum_of_squares_constant_model > 0 else np.nan
        r_squared_adjusted = 1 - (1 - r_squared) * (n - 1) / (n - p) if (n - p) > 0 else np.nan
        return n, design_matrix, betas, r_squared, r_squared_adjusted, design_matrix @ betas
    
class AlignRegressor(ProcessNode):
    outputs = ("aligned_regressor_series",)

    def _run(self, regressor_series : np.ndarray, timeshift_regressor : float, length : int) -> tuple:
        # handled timeshift
        if timeshift_regressor <= 0:
            regressor_series = regressor_series[-timeshift_regressor:]
        else:
            regressor_series = np.concatenate((np.full(timeshift_regressor, np.nan), regressor_series))
        # make sure equal length
        diff = length - len(regressor_series)
        if diff >= 0:
            regressor_series = np.concatenate((regressor_series, np.full(diff, np.nan)))
        else:
            regressor_series = regressor_series[:diff]
        
        return regressor_series, 


class CVRAmplitude(ProcessNode):
    outputs = ("cvr_amplitude",)
    def _run(self, design_matrix : pd.DataFrame, betas : np.ndarray, regressor_baseline : np.ndarray = None) -> tuple:
        mean_features = design_matrix.mean()
        if regressor_baseline is not None:
            mean_features["regressor"] == regressor_baseline
        reg_idx = mean_features.index.get_loc("regressor")
        mean_predict = mean_features @ betas
        return betas[reg_idx] / mean_predict * 100, 

class CVRTimeshift(ProcessNode):
    outputs = ("cvr_timeshift",)
    def _run(self, sample_time : float, timeshift : int, timeshift_reference : int = 0) -> tuple:
        return sample_time * (timeshift - timeshift_reference), 
