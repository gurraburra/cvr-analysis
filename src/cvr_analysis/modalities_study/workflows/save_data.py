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

# %%
##############################################
# create hash
##############################################
def createHashCheckOverride(
                        output_directory, subject, session, task, run, space,
                                voxel_mask, roi_masker, spatial_smoothing_fwhm, 
                                    analysis_start_time, analysis_end_time, min_sample_freq, 
                                        detrend_linear_order, temporal_filter_freq, 
                                            baseline_strategy, use_co2_regressor, confound_regressor_correlation_threshold,
                                                initial_global_align_lower_bound, initial_global_align_upper_bound,
                                                    maxcorr_bipolar, align_regressor_lower_bound, align_regressor_upper_bound, 
                                                        correlation_window, correlation_multi_peak_strategy,
                                                            filter_timeshifts_maxcorr_threshold, filter_timeshifts_size, filter_timeshifts_filter_type,
                                                                refine_regressor_correlation_threshold, refine_regressor_nr_recursions, refine_regressor_explained_variance,
                                                                    ensure_co2_units, 
                                                                        force_run = False): 
    # folder for files
    analysis_name = "cvr-analysis-modalities-" + __version__ 
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

    # do dynamic time warping
    do_dtw = ensure_co2_units and (~use_co2_regressor or refine_regressor_nr_recursions > 0)
   
    # analysis info
    analysis_info = {
        "analysis-name"                             : str(analysis_name),
        "voxel-mask"                                : str(voxel_mask),
        "roi-masker"                                : try_conv(roi_masker, str),
        "spatial-smoothing-fwhm"                    : try_conv(spatial_smoothing_fwhm, float),
        "min-sample-freq"                           : try_conv(min_sample_freq, float),
        "analysis-bounds"                           : try_conv((analysis_start_time, analysis_end_time), float),
        "detrend-linear-order"                      : try_conv(detrend_linear_order, int),
        "temporal-filter-freq"                      : try_conv(temporal_filter_freq, float),
        "baseline-strategy"                         : try_conv(baseline_strategy, str),
        "use-co2-regressor"                         : bool(use_co2_regressor),
        "initial-global-align-bounds"               : try_conv((initial_global_align_lower_bound, initial_global_align_upper_bound), float),
        "align-regressor-bounds"                    : try_conv((align_regressor_lower_bound, align_regressor_upper_bound), float),
        "maxcorr-bipolar"                           : bool(maxcorr_bipolar),
        "correlation-window"                        : try_conv(correlation_window, str),
        "correlation-multi-peak-strategy"           : try_conv(correlation_multi_peak_strategy, str),
        "filter-timeshifts-maxcorr-threshold"       : try_conv(filter_timeshifts_maxcorr_threshold, str),
        "filter-timeshifts-size"                    : try_conv(filter_timeshifts_size, str),
        "filter-timeshifts-filter-type"             : try_conv(filter_timeshifts_filter_type, str),
        "refine-regressor-correlation-threshold"    : try_conv(refine_regressor_correlation_threshold, float),
        "refine-regressor-nr-recursions"            : try_conv(refine_regressor_nr_recursions, int),
        "refine-regressor-explained-variance"       : try_conv(refine_regressor_explained_variance, float),
        "dynamic-time-warping"                      : bool(do_dtw),
        "confound-regressor-correlation-threshold"  : try_conv(confound_regressor_correlation_threshold, float),
    }

    # analysis id
    analysis_id = hashlib.sha1(str(tuple(analysis_info.items())).encode("UTF-8")).hexdigest()[:15]
    # create preamble
    analysis_file = os.path.join(files_folder, f"sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_analys-{analysis_id}_desc-analys_info.json")
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
                    voxel_mask_img, timeseries_masker, bold_tr, co2_event_name,
                        up_sampling_factor, up_sampled_sample_time,
                            # regressor co2 data
                                regressor_preproc_timeseries, regressor_rms, regressor_autocorrelation_timeshifts, regressor_autocorrelation_correlations, 
                                    regressor_aligned_co2_timeseries, co2_rms, co2_autocorrelation_timeshifts, co2_autocorrelation_correlations, 
                                        regressor_co2_timeshift_maxcorr, regressor_co2_maxcorr, regressor_co2_timeshifts, regressor_co2_correlations, 
                                            regressor_co2_beta,
                                                # bold alignement data
                                                reference_regressor_timeshift, align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound,
                                                    bold_preproc_timeseries, bold_timeshift_maxcorr, bold_maxcorr, bold_timeshifts, bold_correlations,
                                                        bold_aligned_regressor_timeseries,
                                                            # bold regression data
                                                            bold_dof, bold_nr_predictors, bold_predictions, bold_r_squared, bold_adjusted_r_squared, bold_tsnr, 
                                                                bold_cvr_amplitude, regression_down_sampled_sample_time,
                                                                    # full output
                                                                    full_output = False) -> tuple:
        
        # make output folder if not exist
        path_folder = Path(os.path.split(analysis_file)[0])
        path_folder.mkdir(parents=True, exist_ok=True)
        # get preable
        preamble = analysis_file.split("desc-")[0]
        # 4D data
        if full_output:
            timeseries_masker.inverse_transform(bold_preproc_timeseries.T).to_filename(preamble + "desc-preproc_bold.nii.gz")
            timeseries_masker.inverse_transform(bold_correlations.T).to_filename(preamble + "desc-correlations_map.nii.gz")
            timeseries_masker.inverse_transform(bold_aligned_regressor_timeseries.T).to_filename(preamble + "desc-alignedRegressor_map.nii.gz")
            timeseries_masker.inverse_transform(bold_predictions.T).to_filename(preamble + "desc-predictions_bold.nii.gz")
        # 3D data
        timeseries_masker.inverse_transform(bold_cvr_amplitude).to_filename(preamble + "desc-cvrAmplitude_map.nii.gz")
        timeseries_masker.inverse_transform(bold_timeshift_maxcorr - reference_regressor_timeshift).to_filename(preamble + "desc-cvrTimeshift_map.nii.gz")
        timeseries_masker.inverse_transform(bold_tsnr).to_filename(preamble + "desc-tsnr_map.nii.gz")
        timeseries_masker.inverse_transform(bold_r_squared).to_filename(preamble + "desc-rSquared_map.nii.gz")
        timeseries_masker.inverse_transform(bold_maxcorr).to_filename(preamble + "desc-maxCorr_map.nii.gz")
        if full_output:
            timeseries_masker.inverse_transform(bold_dof.astype(np.int32)).to_filename(preamble + "desc-dof_map.nii.gz")
            timeseries_masker.inverse_transform(bold_nr_predictors.astype(np.int32)).to_filename(preamble + "desc-nrPredictors_map.nii.gz")
            timeseries_masker.inverse_transform(bold_adjusted_r_squared).to_filename(preamble + "desc-rSquaredAdjusted_map.nii.gz")            
        # 1D data
        if full_output:
            def saveTimeseriesInfo(name, start_time, time_step):
                dict_ = {"start-time" : start_time, "time-step" : time_step}
                with open(name, "w") as file:
                    json.dump(dict_, file, indent='\t')
            # regressor co2
            saveTimeseriesInfo(preamble + "desc-regressorAlignedCO2Series_timeseries.json", 0, regression_down_sampled_sample_time)
            pd.DataFrame(np.vstack((regressor_preproc_timeseries, regressor_aligned_co2_timeseries)).T, columns=["regressor_series", "aligned_co2_series"]).to_csv(preamble + "desc-regressorAlignedCO2Series_timeseries.tsv.gz", sep="\t", index = False, compression="gzip")
            # regressor co2 correlations
            saveTimeseriesInfo(preamble + "desc-regressorCO2Correlations_timeseries.json", regressor_co2_timeshifts[0], up_sampled_sample_time)
            pd.Series(regressor_co2_correlations).to_csv(preamble + "desc-regressorCO2Correlations_timeseries.tsv.gz", sep="\t", index = False, header=False, compression="gzip")
            # regressor autocorrelation
            saveTimeseriesInfo(preamble + "desc-regressorAutocorrelations_timeseries.json", regressor_autocorrelation_timeshifts[0], up_sampled_sample_time)
            pd.Series(regressor_autocorrelation_correlations).to_csv(preamble + "desc-regressorAutocorrelations_timeseries.tsv.gz", sep="\t", index = False, header = False, compression="gzip")
            # co2 autocorrelation
            saveTimeseriesInfo(preamble + "desc-co2Autocorrelations_timeseries.json", co2_autocorrelation_timeshifts[0], up_sampled_sample_time)
            pd.Series(co2_autocorrelation_correlations).to_csv(preamble + "desc-co2Autocorrelations_timeseries.tsv.gz", sep="\t", index = False, header = False, compression="gzip")
        # pd.Series(bold_timeshifts, name="timeshifts").to_csv(preamble + "desc-boldTimeshifts_timeseries.tsv.gz", sep="\t", index = False, compression="gzip")
        if full_output:
            pass
        # timeseries masker
        if isinstance(timeseries_masker, maskers.NiftiMasker):
            timeseries_mask_file = timeseries_masker.mask_img_.get_filename()
        elif isinstance(timeseries_masker, maskers.NiftiLabelsMasker):
            timeseries_mask_file = timeseries_masker.labels_img_.get_filename()
        else:
            raise ValueError(f"Unrecognized timeseries masker: {type(timeseries_masker).__name__}")
        # 1D data
        data_info = {
                    # pre-processing data
                    "subject"                           : subject,
                    "session"                           : session,
                    "task"                              : task,
                    "run"                               : run, 
                    "space"                             : space,
                    "bold-tr"                           : bold_tr,
                    # post-processing data
                    "up-sampling-factor"                : up_sampling_factor,
                    "up-sampled-sample-time"            : up_sampled_sample_time,
                    "voxel-mask-file"                   : voxel_mask_img.get_filename(),
                    "timeseries-masker-file"            : timeseries_mask_file,
                    "co2-event-name"                    : co2_event_name,
                    # regressor co2 data
                    "regressor-co2-timeshift-maxcorr"   : regressor_co2_timeshift_maxcorr,
                    "regressor-co2-maxcorr"             : regressor_co2_maxcorr,
                    "regressor-co2-beta"                : regressor_co2_beta,
                    "regressor-rms"                     : regressor_rms,
                    "co2-rms"                           : co2_rms,
                    # bold alignment data
                    "reference-regressor-timeshift"     : reference_regressor_timeshift, 
                    "align-regressor-absolute-bounds"   : (align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound),
                    "align-regressor-time-step"         : up_sampled_sample_time,
                    "align-regressor-start-time"        : bold_timeshifts[0,0],
                }
        with open(preamble + "desc-data_info.json", "w") as info_file:
            json.dump(data_info, info_file, indent='\t')
        
        # save analysis dict last since this indicate analysis has been completed succesfully
        with open(analysis_file, "w") as info_file:
            json.dump(analysis_dict, info_file, indent='\t')

save_data_node = CustomNode(saveData, description="save data")

conditionally_save_data = ConditionalNode(conditional_input = "save_data", default_condition = True, condition_node_map = {True : save_data_node, False : None}, description = "Save data conditionally")
