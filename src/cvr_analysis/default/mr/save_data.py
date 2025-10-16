# %%
import os
import re
import hashlib
import numpy as np
import pandas as pd
from nilearn import maskers
import json

from process_control import ConditionalNode, CustomNode
from cvr_analysis import __version__
from pathlib import Path

from nilearn.regions import signal_extraction
from nilearn.maskers import NiftiLabelsMasker
# %%
##############################################
# create hash
##############################################
def createHashCheckOverride(
                        output_directory, subject, session, task, run, space,
                                voxel_mask, roi_masker, spatial_smoothing_fwhm, 
                                    initial_time_limit, analysis_start_time, analysis_end_time, min_sample_freq, 
                                        detrend_linear_order, temporal_filter_freq, 
                                            baseline_strategy, regressor, 
                                            # confounds
                                            include_drift_confounds, include_motion_confounds, include_spike_confounds, 
                                            confound_regressor_correlation_threshold,
                                            drift_high_pass, drift_model, drift_order, 
                                            motion_derivatives, motion_powers, 
                                            spike_diff_cutoff, spike_global_cutoff,
                                                global_align_regressor_lower_bound, global_align_regressor_upper_bound,
                                                    maxcorr_bipolar, align_regressor_lower_bound, align_regressor_upper_bound, 
                                                        correlation_phat, correlation_window, correlation_multi_peak_strategy, correlation_peak_threshold,
                                                             filter_timeshifts_size, filter_timeshifts_filter_type, filter_timeshifts_smooth_fwhm,
                                                                refine_regressor_correlation_threshold, refine_regressor_nr_recursions, refine_regressor_explained_variance,
                                                                    do_dtw, dtw_dispersion,
                                                                        force_run = False): 
    # folder for files
    analysis_name = "MRT-CVR-version-" + __version__ 
    files_folder = os.path.join(output_directory, f"sub-{subject}", f"ses-{session}", analysis_name)

    # convert None
    def try_conv(val, type_):
        if val is None:
            return None
        elif isinstance(val, (list, tuple)):
            return tuple(try_conv(v, type_) for v in val)
        else:
            return type_(val)
        
    # check voxel_mask, don't allow 'None' since might lead to ambigious behaviour 
    if voxel_mask is None:
        raise ValueError(f"please specify which voxel mask to be used")
    
    # check if linear order = 0 -> set to = None
    if detrend_linear_order is not None and int(detrend_linear_order) == 0:
        detrend_linear_order = None

    # check if include confounds, if not change threshold
    if not include_motion_confounds:
        confound_regressor_correlation_threshold = None
        motion_derivatives = None
        motion_powers = None

    if not include_drift_confounds:
        drift_high_pass = None
        drift_model = None
        drift_order = None

    if not include_spike_confounds:
        spike_diff_cutoff = None
        spike_global_cutoff = None

    # check if refining regressor, if not change parameters
    if refine_regressor_nr_recursions <= 0:
        refine_regressor_nr_recursions = 0
        refine_regressor_explained_variance = None
        refine_regressor_correlation_threshold = None

    # check if refining regressor, if not change parameters
    if correlation_multi_peak_strategy is None:
        correlation_peak_threshold = None

    # check if filtering timeshifts, if not change parameters
    if filter_timeshifts_filter_type is None:
        filter_timeshifts_size = None

    # check if aligning, if not change parameters
    if align_regressor_lower_bound is not None and align_regressor_upper_bound is not None and align_regressor_lower_bound == align_regressor_upper_bound:
        maxcorr_bipolar = None
        
    # dtw disperion
    if not do_dtw:
        dtw_dispersion = None
    
    # initial time limit
    if analysis_start_time is None and analysis_end_time is None:
        initial_time_limit = None
        
    # analysis info
    analysis_info = {
        "analysis-name"                             : str(analysis_name),
        "voxel-mask"                                : str(voxel_mask),
        "roi-masker"                                : try_conv(roi_masker, str),
        "spatial-smoothing-fwhm"                    : try_conv(spatial_smoothing_fwhm, float),
        "min-sample-freq"                           : try_conv(min_sample_freq, float),
        "initial-time-limit"                        : try_conv(initial_time_limit, bool),
        "analysis-bounds"                           : try_conv((analysis_start_time, analysis_end_time), float),
        "detrend-linear-order"                      : try_conv(detrend_linear_order, int),
        "temporal-filter-freq"                      : try_conv(temporal_filter_freq, float),
        "baseline-strategy"                         : try_conv(baseline_strategy, str),
        "regressor"                                 : str(regressor),
        "global-align-regressor-bounds"             : try_conv((global_align_regressor_lower_bound, global_align_regressor_upper_bound), float),
        "align-regressor-bounds"                    : try_conv((align_regressor_lower_bound, align_regressor_upper_bound), float),
        "maxcorr-bipolar"                           : try_conv(maxcorr_bipolar, bool),
        "correlation-phat"                          : bool(correlation_phat),
        "correlation-window"                        : try_conv(correlation_window, str),
        "correlation-multi-peak-strategy"           : try_conv(correlation_multi_peak_strategy, str),
        "correlation-peak-threshold"                : try_conv(correlation_peak_threshold, float),
        "filter-timeshifts-filter-type"             : try_conv(filter_timeshifts_filter_type, str),
        "filter-timeshifts-smooth-fwhm"             : try_conv(filter_timeshifts_smooth_fwhm, float),
        "filter-timeshifts-size"                    : try_conv(filter_timeshifts_size, int),
        "refine-regressor-nr-recursions"            : try_conv(refine_regressor_nr_recursions, int),
        "refine-regressor-correlation-threshold"    : try_conv(refine_regressor_correlation_threshold, float),
        "refine-regressor-explained-variance"       : try_conv(refine_regressor_explained_variance, float),
        "dtw-to-ensure-regressor-units"             : bool(do_dtw),
        "dtw-dispersion"                            : try_conv(dtw_dispersion, float),
        "include-motion-confounds"                  : bool(include_motion_confounds),
        "include-drift-confounds"                   : bool(include_drift_confounds),
        "include-spike-confounds"                   : bool(include_spike_confounds),
        "drift-high-pass"                           : try_conv(drift_high_pass, float),
        "drift-model"                               : try_conv(drift_model, str),
        "drift-order"                               : try_conv(drift_order, int),
        "motion-derivatives"                        : try_conv(motion_derivatives, bool),
        "motion-powers"                             : try_conv(motion_powers, bool),
        "spike-diff-cutoff"                         : try_conv(spike_diff_cutoff, float),
        "spike-global-cutoff"                       : try_conv(spike_global_cutoff, float),
        "confound-regressor-correlation-threshold"  : try_conv(confound_regressor_correlation_threshold, float),
    }

    # analysis id
    analysis_id = hashlib.sha1(str(tuple(analysis_info.items())).encode("UTF-8")).hexdigest()[:15]
    # create preamble
    def getBStr(var, val):
        if val is not None:
            return f"{var}-{val}_"
        else:
            return ""
    analysis_file = os.path.join(files_folder, f'sub-{subject}_{getBStr("ses",session)}{getBStr("task",task)}{getBStr("run",run)}{getBStr("space",space)}analys-{analysis_id}_desc-analys_info.json')
    # run or dont run
    run_analysis = force_run or not os.path.isfile(analysis_file)

    return run_analysis, analysis_file, analysis_info, analysis_id

create_hash_check_override = CustomNode(createHashCheckOverride, outputs=("run_analysis", "analysis_file", "analysis_dict", "analysis_id"), description="create unique hash and check for old files")


##############################################
# conditionally save data
##############################################
# save data node
def saveData(
            # analysis info data
            analysis_file, analysis_dict,
                # pre processing data
                subject, session, task, run, space,
                    # post-processing data
                    voxel_mask_img, timeseries_masker, bold_tr,
                        up_sampling_factor, up_sampled_sample_time, regressor_units, 
                            # initial global alignement
                            initial_global_regressor_alignment, initial_global_aligned_regressor_timeseries, global_postproc_timeseries,
                                # global regressor signal fit
                                global_regressor_beta, global_regressor_timeshift_maxcorr, global_regressor_maxcorr, global_regressor_timeshifts, global_regressor_correlations, 
                                    global_signal_timeseries, global_aligned_regressor_timeseries, global_regressor_predictions,
                                        # global regressor data
                                        regressor_rms, regressor_autocorrelation_timeshifts, regressor_autocorrelation_correlations, 
                                            global_rms, global_autocorrelation_timeshifts, global_autocorrelation_correlations, 
                                                # bold alignement data
                                                reference_regressor_timeshift, align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound,
                                                    bold_postproc_timeseries, bold_timeshift_maxcorr, bold_maxcorr, bold_timeshifts, bold_correlations,
                                                        bold_aligned_regressor_timeseries,
                                                            # bold regression data
                                                            bold_dof, bold_predictions, bold_r_squared, bold_adjusted_r_squared, bold_standard_error, bold_t_value,
                                                                bold_cvr_amplitude, bold_p_value, regression_sample_time, bold_units,
                                                                # confounds
                                                                regression_confounds_df,
                                                                    # data to save
                                                                    data_to_save = "cvr+tshift+pvalue") -> tuple:
        
        # timeseries info
        def saveTimeseriesInfo(name, start_time, time_step):
            dict_ = {"start-time" : start_time, "time-step" : time_step}
            with open(name, "w") as file:
                json.dump(dict_, file, indent='\t')
        # data to save
        if data_to_save is not None:
            # make output folder if not exist
            path_folder = Path(os.path.split(analysis_file)[0])
            path_folder.mkdir(parents=True, exist_ok=True)
            # get preamble
            preamble = analysis_file.split("desc-")[0]
            # get data to save list
            if isinstance(data_to_save, str):
                # create data list
                data_save_list = data_to_save.lower().split("+")
            elif isinstance(data_to_save, (list,tuple)):
                data_save_list = []
                for d in data_to_save:
                    assert isinstance(d, str)
                    data_save_list.apppend(d.lower())
            else:
                raise ValueError("'data_to_save' needs to be string or list.")
            # check if all data specified in available
            available_data = {"postproc", "xcorrelations", "alignedregressor", "predictions", 
                              "cvr", "tshift", "pvalue", "tvalue", "se", "rsquared", "adjustedrsquared", "maxcorr", "dof", 
                                "initialglobalalignedregressorseries", "globalregressorfit",
                                    "globalregressorxcorrelation", "globalautocorrelation", "regressorautocorrelation", 
                                        "confounds", "mask"}
            non_available_data = set(data_save_list) - available_data 
            if non_available_data:
                raise ValueError(f"Following data is not available: {non_available_data}")
            # save data
            # 4D data
            if "postproc" in data_save_list:
                desc = "postproc"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", 0, up_sampled_sample_time)
                dataToImage(timeseries_masker, bold_postproc_timeseries.T).to_filename(preamble + f"desc-{desc}_bold.nii.gz")
            if "xcorrelation" in data_save_list:
                desc = "xCorrelations"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", bold_timeshifts[0,0], up_sampled_sample_time)
                dataToImage(timeseries_masker, bold_correlations.T).to_filename(preamble + f"desc-{desc}_timeseries.nii.gz")
            if "alignedregressor" in data_save_list:
                desc = "alignedRegressor"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", 0, regression_sample_time)
                dataToImage(timeseries_masker, bold_aligned_regressor_timeseries.T).to_filename(preamble + f"desc-{desc}_timeseries.nii.gz")
            if "predictions" in data_save_list:
                desc = "predictions"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", 0, regression_sample_time)
                dataToImage(timeseries_masker, bold_predictions.T).to_filename(preamble + f"desc-{desc}_bold.nii.gz")
            # 3D data
            if "cvr" in data_save_list:
                dataToImage(timeseries_masker, bold_cvr_amplitude).to_filename(preamble + "desc-cvr_map.nii.gz")
            if "tshift" in data_save_list:
                dataToImage(timeseries_masker, bold_timeshift_maxcorr).to_filename(preamble + "desc-tshift_map.nii.gz")
            if "pvalue" in data_save_list:
                dataToImage(timeseries_masker, bold_p_value).to_filename(preamble + "desc-pValue_map.nii.gz")
            if "tvalue" in data_save_list:
                dataToImage(timeseries_masker, bold_t_value).to_filename(preamble + "desc-tValue_map.nii.gz")
            if "se" in data_save_list:
                dataToImage(timeseries_masker, bold_standard_error).to_filename(preamble + "desc-se_map.nii.gz")
            if "rsquared" in data_save_list:
                dataToImage(timeseries_masker, bold_r_squared).to_filename(preamble + "desc-rSquared_map.nii.gz")
            if "adjustedrsquared" in data_save_list:
                dataToImage(timeseries_masker, bold_adjusted_r_squared).to_filename(preamble + "desc-adjustedRSquared_map.nii.gz")            
            if "maxcorr" in data_save_list:
                dataToImage(timeseries_masker, bold_maxcorr).to_filename(preamble + "desc-maxCorr_map.nii.gz")
            if "dof" in data_save_list:
                dataToImage(timeseries_masker, bold_dof.astype(np.int32)).to_filename(preamble + "desc-dof_map.nii.gz")
            if "confounds" in data_save_list and regression_confounds_df is not None:
                regression_confounds_df.to_csv(preamble + "desc-confounds_timeseries.tsv.gz", sep="\t", index = False, compression="gzip")
            if "mask" in data_save_list:
                voxel_mask_img.to_filename(preamble + "desc-voxel_mask.nii.gz")
            # 1D data
            if "initialglobalalignedregressorseries" in data_save_list:
                # global regressor
                desc = "initialGlobalAlignedRegressorSeries"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", 0, up_sampled_sample_time)
                pd.DataFrame(np.vstack((global_postproc_timeseries, initial_global_aligned_regressor_timeseries)).T, columns=["global_series", "aligned_regressor_series"]).to_csv(preamble + f"desc-{desc}_timeseries.tsv.gz", sep="\t", index = False, compression="gzip")
            if "globalregressorfit" in data_save_list:
                # global regressor
                desc = "globalRegressorFit"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", 0, regression_sample_time)
                pd.DataFrame(np.vstack((global_signal_timeseries, global_aligned_regressor_timeseries, global_regressor_predictions)).T, columns=["global_series", "aligned_regressor_series", "predictions"]).to_csv(preamble + f"desc-{desc}_timeseries.tsv.gz", sep="\t", index = False, compression="gzip")
            if "globalregressorxcorrelation" in data_save_list: 
                # global regressor correlations
                desc = "globalRegressorrXCorrelation"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", global_regressor_timeshifts[0], up_sampled_sample_time)
                pd.Series(global_regressor_correlations).to_csv(preamble + f"desc-{desc}_timeseries.tsv.gz", sep="\t", index = False, header=False, compression="gzip")
            if "regressorautocorrelation" in data_save_list: 
                # regressor autocorrelation
                desc = "regressorAutocorrelation"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", regressor_autocorrelation_timeshifts[0], up_sampled_sample_time)
                pd.Series(regressor_autocorrelation_correlations).to_csv(preamble + f"desc-{desc}_timeseries.tsv.gz", sep="\t", index = False, header = False, compression="gzip")
            if "globalautocorrelation" in data_save_list:
                # global autocorrelation
                desc = "globalAutocorrelation"
                saveTimeseriesInfo(preamble + f"desc-{desc}_timeseries.json", global_autocorrelation_timeshifts[0], up_sampled_sample_time)
                pd.Series(global_autocorrelation_correlations).to_csv(preamble + f"desc-{desc}_timeseries.tsv.gz", sep="\t", index = False, header = False, compression="gzip")
            # timeseries masker
            if isinstance(timeseries_masker, maskers.NiftiMasker):
                timeseries_mask_file = timeseries_masker.mask_img_.get_filename()
            elif isinstance(timeseries_masker, maskers.NiftiLabelsMasker):
                timeseries_mask_file = timeseries_masker.labels_img_.get_filename()
            else:
                raise ValueError(f"Unrecognized timeseries masker: {type(timeseries_masker).__name__}")
            # 1D data
            if regression_confounds_df is not None:
                confounds_names = list(regression_confounds_df.columns)
            else:
                confounds_names = None
            data_info = {
                        # pre-processing data
                        "subject"                               : subject,
                        "session"                               : session,
                        "task"                                  : task,
                        "run"                                   : run, 
                        "space"                                 : space,
                        "bold-tr"                               : bold_tr,
                        # post-processing data
                        "up-sampling-factor"                    : up_sampling_factor,
                        "up-sampled-sample-time"                : up_sampled_sample_time,
                        "regression-sample-time"                : regression_sample_time,
                        "voxel-mask-file"                       : voxel_mask_img.get_filename(),
                        "timeseries-masker-file"                : timeseries_mask_file,
                        # global regressor data
                        "regressor-units"                       : regressor_units,
                        "bold-units"                            : bold_units,
                        "initial-global-regressor-alignment"    : initial_global_regressor_alignment,
                        "global-regressor-timeshift-maxcorr"    : global_regressor_timeshift_maxcorr,
                        "global-regressor-maxcorr"              : global_regressor_maxcorr,
                        "global-regressor-beta"                 : global_regressor_beta,
                        "regressor-rms"                         : regressor_rms,
                        "global-rms"                            : global_rms,
                        # bold alignment data
                        "reference-regressor-timeshift"         : reference_regressor_timeshift, 
                        "align-regressor-absolute-bounds"       : (align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound),
                        # confound names
                        "confounds-names"                       : confounds_names,
                    }
            with open(preamble + "desc-data_info.json", "w") as info_file:
                json.dump(data_info, info_file, indent='\t')
            
            # save analysis dict last since this indicate analysis has been completed succesfully
            with open(analysis_file, "w") as info_file:
                json.dump(analysis_dict, info_file, indent='\t')

save_data_node = CustomNode(saveData, description="save data")

# conditionally_save_data = ConditionalNode(conditional_input = "save_data", default_condition = True, condition_node_map = {True : save_data_node, False : None}, description = "Save data conditionally")

#%%
def dataToImage(timeseries_masker, data):
    # this function avoid the nan value check in NiftiLabelsMasker.inverse_transform
    if isinstance(timeseries_masker, NiftiLabelsMasker):
        return signal_extraction.signals_to_img_labels(data, 
                                                    timeseries_masker.labels_img_,
                                                    timeseries_masker.mask_img_,
                                                    background_label=timeseries_masker.background_label,
        )
    else:
        return timeseries_masker.inverse_transform(data)