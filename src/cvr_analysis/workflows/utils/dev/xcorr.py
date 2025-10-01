#%%
from scipy.fft import rfft, irfft, next_fast_len
import numpy as np
from process_control import ProcessNode, ProcessWorkflow, ValueNode, CustomNode, ConditionalNode, IteratingNode
import scipy.signal as sc_signal
from scipy.ndimage import vectorized_filter
from nilearn.maskers import NiftiMasker
import nibabel as nib
import pandas as pd
from nilearn.glm.first_level import run_glm
from scipy import stats
import warnings
from multiprocess.sharedctypes import RawArray
from cvr_analysis.workflows.utils.dev.helper_functions import removeNan

# %%
class XCorr(ProcessNode):
    """
    Calculates cross-correlation between signals and reference and and return the shift of
    reference relative to signals: positive shift means reference have beed shifted 
    to the right, i.e leads signals. Assume time dimension is the first dimension.
    """

    outputs = ("timeshifts", "correlations")
    def _run(self, signals : np.ndarray, reference : np.ndarray, time_step : float, lower_limit : float = None, upper_limit : float = None, phat : bool = True, window : str = "hamming"):
        # check dimensions
        if signals.ndim == 1:
            c_signals = signals[:,None]
        elif signals.ndim == 2:
            c_signals = signals
        else:
            raise ValueError("'signals' can only have 1 or 2 dimensions")
        assert reference.ndim == 1, "'reference' can only have 1 dimension"
        reference = reference[:,None] # will make things easier if signals and reference have same #dimensions
        # handle leading and trailing nan values
        # signals
        signals_nan_removed, signals_leading_nan, signals_trailing_nan = removeNan(c_signals)
        # reference
        reference_nan_removed, reference_leading_nan, reference_trailing_nan = removeNan(reference)
        # check if all nan
        if signals_nan_removed.size == 0 or reference_nan_removed.size == 0:
            raise ValueError("To many nan values")
        # length
        len_signals_nan_removed = signals_nan_removed.shape[0]
        len_reference_nan_removed = reference_nan_removed.shape[0]
        # check window
        if window is None:
            window_signals = np.ones((len_signals_nan_removed, 1))
            window_reference = np.ones((len_reference_nan_removed, 1))
        else:
            window_signals = sc_signal.get_window(window, len_signals_nan_removed)[:,None]
            window_reference = sc_signal.get_window(window, len_reference_nan_removed)[:,None]
        # applying a window (basically because RapidTide suggest so)
        windowed_signals = signals_nan_removed * window_signals
        windowed_reference = reference_nan_removed * window_reference
        # calculate cross-corr
        correlations_nan_removed = self.xcorr(windowed_signals, windowed_reference, axis = 0, phat=phat, fast_len=True)
        # add nan values
        correlations = np.concatenate(
                (
                    np.full((signals_leading_nan + reference_trailing_nan, correlations_nan_removed.shape[1]), np.nan), 
                    correlations_nan_removed,
                    np.full((signals_trailing_nan + reference_leading_nan, correlations_nan_removed.shape[1]), np.nan)
                ), axis=0
            )
        # timeshifts
        timeshifts = np.arange(-reference.shape[0]+1, c_signals.shape[0], 1) * time_step
        # mask
        mask = np.full_like(timeshifts, True, dtype = bool)
        if lower_limit is not None:
            mask[timeshifts < lower_limit] = False
        if upper_limit is not None:
            mask[timeshifts > upper_limit] = False
        # check mask
        if not np.any(mask):
            if lower_limit is not None and upper_limit is not None:
                # pick value closet to the limits
                mask[np.argmin(np.abs(timeshifts - (lower_limit + upper_limit) / 2))] = True
            else:
                raise ValueError("Incorrect limits specified.")
        # mask timeshifts
        timeshifts_masked = timeshifts[mask]
        # bound correlations
        correlations_masked = correlations[mask]
        if signals.ndim == 1:
            return timeshifts_masked, correlations_masked[:,0]
        else:
            return timeshifts_masked, correlations_masked
    
    def xcorr(self, s1, s2, axis = 0, fast_len = False, phat = False):
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        assert s1.ndim == s2.ndim, "Arrays have different number of dimension."
        length = s1.shape[axis] + s2.shape[axis] - 1
        if fast_len:
            f_length = next_fast_len(length)
        else:
            f_length = length
        f_s1 = rfft(s1, f_length, axis = axis)
        f_s2 = rfft(np.flip(s2, axis = axis), f_length, axis = axis)

        f_s = f_s1 * f_s2
        if phat:
            weight = np.abs(f_s)
            weight[weight < 1e-6] = 1e-6
            f_s = f_s / weight
        corr = irfft(f_s, f_length, axis = axis)
        # if fast len
        if fast_len:
            corr = corr.take(indices = range(length), axis = axis)
        return corr
