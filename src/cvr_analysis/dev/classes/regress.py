#%%
import numpy as np
from process_control import ProcessNode
import pandas as pd
from nilearn.glm.first_level import run_glm
from scipy import stats

# %%
class RegressNilearn(ProcessNode):
    """
    Performs linear regression between regressor data och bold data.
    """
    # outputs = ("labels", "result")
    outputs = ("design_matrix", "betas", "predictions", "dof", "r_squared", "adjusted_r_squared", "regressor_beta", "regressor_se", "regressor_t", "regressor_p")

    def _run(self, dv_ts : np.ndarray, regressor_ts : np.ndarray, confounds_df : pd.DataFrame = None, confound_regressor_correlation_threshold : float = None, regressor_delay : float = 0, regressor_time_step : float = 1.0, regressor_down_sampling_factor : int = 1, noise_model : str = "ols") -> tuple:
        # check regressor
        if regressor_ts is None:
            regressor_ts = pd.DataFrame([], index = range(dv_ts.shape[0]))
        else:
            # delay
            delay_steps = int(regressor_delay // regressor_time_step)
            if delay_steps >= 0:
                regressor_ts = np.concat(([np.nan] * delay_steps, regressor_ts))
            else:    
                regressor_ts = regressor_ts[-delay_steps:]
            # down_sample
            regressor_ts = regressor_ts[::regressor_down_sampling_factor]
            # make sure equal length
            diff_len = dv_ts.shape[0] - regressor_ts.shape[0]
            if diff_len >= 0:
                regressor_ts = np.concat((regressor_ts, [np.nan] * diff_len))
            else:
                regressor_ts = regressor_ts[:diff_len]
            # make into series object
            regressor_ts = pd.Series(regressor_ts, name = "regressor")
        # check confounds_df
        if confounds_df is None:
            confounds_df = pd.DataFrame([], index = range(dv_ts.shape[0]))
        else:
            confounds_df = pd.DataFrame(confounds_df)
        # make sure columns are string
        confounds_df.columns = confounds_df.columns.astype(str)
        # assert len
        assert dv_ts.shape[0] == regressor_ts.shape[0], "Regressor and dependent variables have unequal length."
        assert dv_ts.shape[0] == confounds_df.shape[0], "Confounds and dependent variables have unequal length."
        # create design matrix
        design_matrix = pd.concat((regressor_ts, confounds_df.reset_index(drop=True)), axis = 1)
        if design_matrix.empty:
            raise ValueError("No independent variables provided.")
        # add constant
        if "constant" not in design_matrix:
            design_matrix["constant"] = 1
        # find nan rows
        nan_entries = design_matrix.isna().any(axis=1)
        # check if no valid entries
        nan_arr = np.full(dv_ts.shape[1], np.nan)
        if np.all(nan_entries):
            # return all nans
            return design_matrix, np.full((design_matrix.shape[1], dv_ts.shape[1]) , np.nan), np.full_like(dv_ts, np.nan), 0, nan_arr, nan_arr, nan_arr, nan_arr, nan_arr, nan_arr
        non_nan_dm = design_matrix[~nan_entries]
        non_nan_dv = dv_ts[~nan_entries]
        # threshold confunds
        if confound_regressor_correlation_threshold is not None and not confounds_df.empty and not regressor_ts.empty:
            # filter out protected non_protected_confounds
            protected_confound_names = ["constant"]
            protected_confound_names.extend(list(filter(lambda x : x.startswith("spike"), confounds_df.columns)))
            protected_confound_names.extend(list(filter(lambda x : x.startswith("drift"), confounds_df.columns)))
            non_protected_confound_names = confounds_df.columns[~confounds_df.columns.isin(protected_confound_names)]
            if non_protected_confound_names.empty:
                non_protected_confounds_above_thr = []
            else:
                # correlate
                non_protected_confound_corr = np.corrcoef(non_nan_dm["regressor"], non_nan_dm[non_protected_confound_names], rowvar=False)[0, 1:]
                # threshold
                non_protected_confounds_above_thr = non_protected_confound_names[np.abs(non_protected_confound_corr) > confound_regressor_correlation_threshold]
            # drop
            design_matrix = design_matrix.drop(columns = non_protected_confounds_above_thr)
            non_nan_dm = non_nan_dm.drop(columns = non_protected_confounds_above_thr)
        # get dimensions
        n, p = non_nan_dm.shape
        dof = n - p
        # regress
        glm_labels, glm_result = run_glm(non_nan_dv, non_nan_dm.to_numpy(), noise_model=noise_model)
        # compute_contrast(glm_labels, glm_result, con_val=[0,1]).stat()
        betas, r_squared = self._glmResultToArr(glm_labels, glm_result, "theta", p, False),  self._glmResultToArr(glm_labels, glm_result, "r_square", None, False)
        pred = design_matrix.to_numpy() @ betas
        # adjusted r-squared
        if dof - 1 > 0:
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (dof - 1)
        else:
            adjusted_r_squared = np.full_like(r_squared, np.nan)
        # regressor beta, t- and p-value
        if not regressor_ts.empty:
            # get regressor stats
            reg_idx = design_matrix.columns.get_loc("regressor")
            regressor_beta, regressor_se = betas[reg_idx], np.sqrt(self._glmResultToArr(glm_labels, glm_result, "vcov", None, True, column = reg_idx))
            regressor_t = regressor_beta / regressor_se
            regressor_p = 2 * stats.t.sf(np.abs(regressor_t), dof)
            # regressor_stats = np.vstack((regressor_beta, regressor_se, regressor_t, regressor_p))
        else:
            regressor_beta, regressor_se, regressor_t, regressor_p = None, None, None, None, None
        # return
        return design_matrix, betas, pred, dof, r_squared, adjusted_r_squared, regressor_beta, regressor_se, regressor_t, regressor_p
    
    @staticmethod
    def _glmResultToArr(labels, result_dict, variable : tuple, size : tuple, call : bool, *args, **kw_args):
        s = size if size is not None else 1
        arr = np.empty((s, len(labels)))
        for label_, glm_result in result_dict.items():
            label_mask = labels == label_
            if call:
                arr[:,label_mask] = getattr(glm_result, variable)(*args, **kw_args)
            else:
                arr[:,label_mask] = getattr(glm_result, variable)
        if size is None:
            return arr[0]
        else:
            return arr
