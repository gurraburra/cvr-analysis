#%%
import numpy as np
from process_control import ProcessNode
import scipy.signal as sc_signal
from scipy.ndimage import vectorized_filter
from nilearn.maskers import NiftiMasker
import nibabel as nib
import warnings
from cvr_analysis.workflows.utils.dev.helper_functions import removeNan

# %%
class FindPeaks(ProcessNode):
    outputs = ("peaks_array",)
    def _run(self, y : np.ndarray, bipolar : bool = True, min_height = 0.5, max_nr_peaks = 10):
        # check dimensions
        if y.ndim == 1:
            p_y = y[:,None]
        elif y.ndim == 2:
            p_y = y
        else:
            raise ValueError("'y' must have 1 or 2 #dimensions.")
        
        # peak finding values (before adding nan)
        if bipolar:
            p_y_polar = np.abs(p_y)
        else:
            p_y_polar = p_y
        
        def _find_peaks(arr, max_nr_peaks, offset, height):
            # Find all peaks
            peaks, _ = sc_signal.find_peaks(arr, height=height)
            # Get their heights
            heights = arr[peaks]
            # Sort peaks by height descending
            sorted_indices = np.argsort(heights)[::-1]
            # fix lenght
            peaks_fix_len = np.full(max_nr_peaks, np.nan)
            n = min(len(peaks), max_nr_peaks)
            peaks_fix_len[:n] = peaks[sorted_indices[:n]]
            # return
            return peaks_fix_len + offset
        
        # peak finding algorithm
        p_y_polar_nan_removed, p_y_polar_leading_nan, _ = removeNan(p_y_polar)
        peaks_array = np.apply_along_axis(_find_peaks, 0, p_y_polar_nan_removed, max_nr_peaks, p_y_polar_leading_nan, min_height)
        # check dimension
        if y.ndim == 1:
            return peaks_array[:,0], 
        else:
            return peaks_array, 


# %%
class FilterPeaks(ProcessNode):
    outputs = ("filtered_peaks",)
    def _run(self, peaks_array : np.ndarray, timeseries_masker : NiftiMasker = None, filter_type : str = "median", kernel_size : int = 3):
        # check filter type
        if filter_type is None:
            # if filtered type is none, return highest peak (which is the first one given they are sorted)
            # only option possible for single dimension peaks_array
            if peaks_array.ndim == 1:
                return peaks_array[0].astype(int),
            elif peaks_array.ndim == 2:
                p_max = peaks_array[0,:]
                nanvalues = np.isnan(p_max)
                if np.all(nanvalues):
                    raise RuntimeError("No peaks available")
                med = np.nanmedian(p_max)
                return np.where(nanvalues, med, p_max).astype(int),
            else:
                raise ValueError("'peaks_array' must have 1 or 2 #dimensions.")
        else: # filter -> check peaks_array is two dimensional
            assert peaks_array.ndim == 2, "'peaks_array' must have 2 #dimensions when filtering"
            assert timeseries_masker is not None, "must provide 'timeseries_masker' when filtering"
            if filter_type == "median":
                filter_func = np.nanmedian
            elif filter_type == "mean":
                filter_func = np.nanmean
            else:
                raise ValueError("'filter_type' can only one out of the following values: median, mean or None")
            # raise ValueError("'filter_type' can only one out of the following values: median, mean, min, max")
        # temporay cahnge smoothing
        old_smooth_fwhm = timeseries_masker.smoothing_fwhm 
        timeseries_masker.smoothing_fwhm = None
        # convert peaks into img
        peaks_array_img = timeseries_masker.inverse_transform(peaks_array).get_fdata()
        # get mask img
        mask_img = timeseries_masker.mask_img_.get_fdata() > 0.1
        # create peak type mask
        nr_peaks = np.sum(~np.isnan(peaks_array_img),axis = -1)
        single_peak_mask = (nr_peaks == 1) * mask_img
        multi_peak_mask = (nr_peaks > 1) * mask_img
        zero_peak_mask = (nr_peaks == 0) * mask_img

        # create initial peak img with all nans
        peak_img = np.full(peaks_array_img.shape[:-1], np.nan)
        # create new img with single_peak values added
        new_peak_img = peak_img.copy()
        new_peak_img[single_peak_mask] = peaks_array_img[single_peak_mask,0]

        # surpress warning when all nan values encounted
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
            # iterate until convergation
            while not np.array_equal(peak_img, new_peak_img, equal_nan=True):
                peak_img = new_peak_img
                
                filtered_peak_img = vectorized_filter(peak_img, filter_func, size = kernel_size)
                new_peak_img = peak_img.copy()
                new_peak_img[zero_peak_mask] = filtered_peak_img[zero_peak_mask]


                multi_peak_mask_tmp = multi_peak_mask & ~np.isnan(filtered_peak_img)
                closest_multi_peak = np.nanargmin(np.abs(peaks_array_img[multi_peak_mask_tmp] - filtered_peak_img[multi_peak_mask_tmp][...,None]), axis = -1)
                new_peak_img[multi_peak_mask_tmp] = peaks_array_img[multi_peak_mask_tmp, closest_multi_peak]

        filtered_peaks = np.round(timeseries_masker.transform(nib.Nifti1Image(new_peak_img, timeseries_masker.mask_img_.affine))).astype(int)
        # change back smoothing 
        timeseries_masker.smoothing_fwhm = old_smooth_fwhm
        return filtered_peaks, 

