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
    outputs = ("dof", "nr_predictors", "design_matrix", "betas", "r_squared",  "r_squared_adjusted", "predictions", "nr_removed_colinear")

    def _run(self, bold_series : np.ndarray, regressor_series : np.ndarray = None, confounds_df : pd.DataFrame = None, colinear : str = None, colinear_thr : float = 0) -> tuple:
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
        # handle collinearity
        if colinear is not None:
            colinear_mask = self._get_colinear_confounds(non_nan_dm, method=colinear, threshold=colinear_thr)
            design_matrix = design_matrix.loc[:, ~colinear_mask]
            non_nan_dm = non_nan_dm.loc[:, ~colinear_mask]
            nr_colinear = np.sum(colinear_mask)
        else:
            nr_colinear = 0
        # get dimensions
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
        return n, p, design_matrix, betas, r_squared, r_squared_adjusted, design_matrix @ betas, nr_colinear
    
    def _get_colinear_confounds(self, design_matrix : pd.DataFrame, method : str = "corr", threshold : float = 0):
        """
        removes colineary using vector inflation factor (method = 'vif') or correlations with regressor (method = 'corr')
        """
        if method == "vif":
            colinear_cofounds = []
            confounds = design_matrix.columns.to_list()
            # constant and regressor is not considered confounds
            if "constant" in confounds:
                confounds.remove("constant")
            if "regressor" in confounds:
                confounds.remove("regressor")
            # loop through while maximimum vif is above threshold
            while True:
                # chreate list of vifs
                vifs = [0]*len(confounds)
                # calculate vif
                for i, c in enumerate(confounds):
                    # remove confound idx
                    X,y  = design_matrix[confounds].drop(columns=c), design_matrix[c]
                    vifs[i] = 1 / (1 - lm.LinearRegression().fit(X,y).score(X,y))
                # remove max if above threshold
                if len(vifs) > 0 and np.max(vifs) > threshold:
                    conf_max = confounds[np.argmax(vifs)]
                    confounds.remove(conf_max)
                    colinear_cofounds.append(conf_max)
                    continue
                else:
                    # if no vif greater than threshold
                    break
            # return mask for columns
            return np.array([True if c in colinear_cofounds else False for c in design_matrix.columns])
        elif method == 'corr':
            confounds = design_matrix.columns.to_list()
            # constant and regressor is not considered confounds
            if "constant" in confounds:
                confounds.remove("constant")
            if "regressor" in confounds:
                confounds.remove("regressor")
            else:
                return [True]*design_matrix.shape[1]
            
            confounds_mat = design_matrix[confounds].values.T
            regressor_vec = design_matrix["regressor"].values[None,:] 
            
            # Rowwise mean of input arrays & subtract from input arrays themeselves
            A_mA = confounds_mat - confounds_mat.mean(1)[:, None]
            B_mB = regressor_vec - regressor_vec.mean(1)[:, None]

            # Sum of squares across rows
            ssA = (A_mA**2).sum(1)
            ssB = (B_mB**2).sum(1)

            # Finally get corr coeff
            corr_abs = np.abs(np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None, :])))
            confounds_mask = corr_abs > threshold
            colinear_cofounds = [c for i,c in enumerate(confounds) if confounds_mask[i]]
            # return mask for columns
            return np.array([True if c in colinear_cofounds else False for c in design_matrix.columns])
        else:
            raise ValueError(f"Invalid method for removal of colinearity given: {method}.")
    
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
