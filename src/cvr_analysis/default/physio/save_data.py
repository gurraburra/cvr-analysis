# %%
import os
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
                        # bids
                        output_directory, subject, session, task, run, physio_recording, data_type,
                            initial_time_limit, analysis_start_time, analysis_end_time, min_sample_freq, 
                                detrend_linear_order, temporal_filter_freq, 
                                    baseline_strategy, regressor, psc_regressor, psc_physio,
                                        refine_regressor_correlation_threshold, refine_regressor_explained_variance, refine_regressor_nr_recursions,
                                            do_dtw, dtw_dispersion,
                                                # confounds
                                                include_drift_confounds, include_spike_confounds, 
                                                    drift_high_pass, drift_model, drift_order, 
                                                        spike_diff_cutoff, spike_global_cutoff,
                                                            # align
                                                            global_align_regressor_lower_bound, global_align_regressor_upper_bound,
                                                                maxcorr_bipolar, align_regressor_lower_bound, align_regressor_upper_bound, 
                                                                    correlation_phat, correlation_window, correlation_multi_peak_strategy, correlation_peak_threshold,
                                                                        force_run = False): 
    # folder for files
    if data_type is None:
        raise ValueError("'data_type' need to be not 'None'")
    files_folder = os.path.join(output_directory, f"sub-{subject}", f"ses-{session}", data_type)
    analysis_name = "Physio-CVR-version-" + __version__ 

    # convert None
    def try_conv(val, type_):
        if val is None:
            return None
        elif isinstance(val, (list, tuple)):
            return tuple(try_conv(v, type_) for v in val)
        else:
            return type_(val)
    
    # check if linear order = 0 -> set to = None
    if detrend_linear_order is not None and int(detrend_linear_order) == 0:
        detrend_linear_order = None

    # confounds
    if not include_drift_confounds:
        drift_high_pass = None
        drift_model = None
        drift_order = None

    if not include_spike_confounds:
        spike_diff_cutoff = None
        spike_global_cutoff = None

    # check if refining regressor, if not change parameters
    if correlation_multi_peak_strategy is None:
        correlation_peak_threshold = None

    # check if aligning, if not change parameters
    if align_regressor_lower_bound is not None and align_regressor_upper_bound is not None and align_regressor_lower_bound == align_regressor_upper_bound:
        maxcorr_bipolar = None

    # initial time limit
    if analysis_start_time is None and analysis_end_time is None:
        initial_time_limit = None

    # check if refining regressor, if not change parameters
    if refine_regressor_nr_recursions <= 0:
        refine_regressor_nr_recursions = 0
        refine_regressor_explained_variance = None
        refine_regressor_correlation_threshold = None

    # dtw disperion
    if not do_dtw:
        dtw_dispersion = None
   
    # analysis info
    analysis_info = {
        "analysis-name"                             : str(analysis_name),
        "min-sample-freq"                           : try_conv(min_sample_freq, float),
        "initial-time-limit"                        : try_conv(initial_time_limit, bool),
        "analysis-bounds"                           : try_conv((analysis_start_time, analysis_end_time), float),
        "detrend-linear-order"                      : try_conv(detrend_linear_order, int),
        "temporal-filter-freq"                      : try_conv(temporal_filter_freq, float),
        "psc-physio"                                : try_conv(psc_physio, bool),
        "baseline-strategy"                         : try_conv(baseline_strategy, str),
        "regressor"                                 : str(regressor),
        "psc-regressor"                             : try_conv(psc_regressor, bool),
        "refine-regressor-nr-recursions"            : try_conv(refine_regressor_nr_recursions, int),
        "refine-regressor-correlation-threshold"    : try_conv(refine_regressor_correlation_threshold, float),
        "refine-regressor-explained-variance"       : try_conv(refine_regressor_explained_variance, float),
        "dtw-to-ensure-regressor-unit"              : bool(do_dtw),
        "dtw-dispersion"                            : try_conv(dtw_dispersion, float),
        "global-align-regressor-bounds"             : try_conv((global_align_regressor_lower_bound, global_align_regressor_upper_bound), float),
        "align-regressor-bounds"                    : try_conv((align_regressor_lower_bound, align_regressor_upper_bound), float),
        "maxcorr-bipolar"                           : try_conv(maxcorr_bipolar, bool),
        "correlation-phat"                          : bool(correlation_phat),
        "correlation-window"                        : try_conv(correlation_window, str),
        "correlation-multi-peak-strategy"           : try_conv(correlation_multi_peak_strategy, str),
        "correlation-peak-threshold"                : try_conv(correlation_peak_threshold, float),
        "include-drift-confounds"                   : bool(include_drift_confounds),
        "include-spike-confounds"                   : bool(include_spike_confounds),
        "drift-high-pass"                           : try_conv(drift_high_pass, float),
        "drift-model"                               : try_conv(drift_model, str),
        "drift-order"                               : try_conv(drift_order, int),
        "spike-diff-cutoff"                         : try_conv(spike_diff_cutoff, float),
        "spike-global-cutoff"                       : try_conv(spike_global_cutoff, float),
    }

    # analysis id
    analysis_id = hashlib.sha1(str(tuple(analysis_info.items())).encode("UTF-8")).hexdigest()[:15]
    # create preamble
    def getBStr(var, val):
        if val is not None:
            return f"{var}-{val}_"
        else:
            return ""
    analysis_file = os.path.join(files_folder, f'sub-{subject}_{getBStr("ses",session)}{getBStr("task",task)}{getBStr("run",run)}{getBStr("recording",physio_recording)}analys-{analysis_id}_desc-analys_info.json')
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
                subject, session, task, run, physio_variables,
                    # post-processing data
                    physio_tr, up_sampling_factor, up_sampled_sample_time, regressor_unit, regressor_baseline,
                        # initial global alignement
                        initial_global_regressor_alignment, initial_global_aligned_regressor_timeseries, global_postproc_timeseries, global_baseline,
                            # global regressor signal fit
                            global_regressor_cvr_amplitude, global_regressor_timeshift_maxcorr, global_regressor_maxcorr, global_regressor_timeshifts, global_regressor_correlations,
                                global_regressor_p_value, global_regressor_standard_error, global_regressor_t_value, global_regressor_dof, global_regressor_r_squared, global_regressor_adjusted_r_squared,
                                    down_sampled_global_postproc_timeseries, down_sampled_global_aligned_regressor_timeseries, down_sampled_global_regressor_predictions,
                                        # physio alignement data
                                        reference_regressor_timeshift, align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound,
                                            physio_postproc_timeseries, physio_timeshift_maxcorr, physio_maxcorr, physio_timeshifts, physio_correlations,
                                                physio_aligned_regressor_timeseries,
                                                    # physio regression data
                                                    physio_dof, physio_predictions, physio_r_squared, physio_adjusted_r_squared, physio_standard_error, physio_t_value,
                                                        physio_cvr_amplitude, physio_p_value, regression_sample_time, physio_unit, physio_baseline,
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
            # get preable
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
                                "initialglobalalignedregressorseries", "globalregressorfit", "globalregressorxcorrelation",
                                        "confounds"}
            non_available_data = set(data_save_list) - available_data 
            if non_available_data:
                raise ValueError(f"Following data is not available: {non_available_data}")
            # 2D data
            if "postproc" in data_save_list:
                fname = preamble + "desc-postproc_physio"
                saveTimeseriesInfo(fname + ".json", 0, up_sampled_sample_time)
                pd.DataFrame(physio_postproc_timeseries.T, columns=physio_variables).to_csv(fname + ".tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            if "predictions" in data_save_list:
                fname = preamble + "desc-predictions_physio"
                saveTimeseriesInfo(fname + ".json", 0, regression_sample_time)
                pd.DataFrame(physio_predictions.T, columns=physio_variables).to_csv(fname + ".tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            if "alignedregressor":
                fname = preamble + "desc-alignedRegressor_timeseries"
                saveTimeseriesInfo(fname + ".json", 0, regression_sample_time)
                pd.DataFrame(physio_aligned_regressor_timeseries.T, columns=physio_variables).to_csv(fname + ".tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            if "xcorrelations" in data_save_list:
                fname = preamble + "desc-xCorrelations_timeseries"
                saveTimeseriesInfo(fname + ".json", physio_timeshifts[0,0], regression_sample_time)
                pd.DataFrame(physio_correlations.T, columns=physio_variables).to_csv(fname + ".tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            # dataframe data
            df_dict = {
                "signal" : tuple(physio_variables) + ("global", ),
                "unit" : [physio_unit]*(len(physio_variables) + 1),
                "mean" : np.append(physio_postproc_timeseries.mean(axis = 1), global_postproc_timeseries.mean()),
                "max" : np.append(physio_postproc_timeseries.max(axis = 1), global_postproc_timeseries.max()),
                "min" : np.append(physio_postproc_timeseries.min(axis = 1), global_postproc_timeseries.min()),
                "std" : np.append(physio_postproc_timeseries.std(axis = 1), global_postproc_timeseries.std()),
                "init_baseline" : np.append(physio_baseline, global_baseline),
                "cvr" : np.append(physio_cvr_amplitude,global_regressor_cvr_amplitude),
                "timeshift" : np.append(physio_timeshift_maxcorr, global_regressor_timeshift_maxcorr), 
                "maxcorr" : np.append(physio_maxcorr, global_regressor_maxcorr),
                "p" : np.append(physio_p_value, global_regressor_p_value),
                "se" : np.append(physio_standard_error, global_regressor_standard_error),
                "t" : np.append(physio_t_value, global_regressor_t_value),
                "dof" : np.append(physio_dof, global_regressor_dof),
                "r2" : np.append(physio_r_squared, global_regressor_r_squared),
                "adj_r2" : np.append(physio_adjusted_r_squared, global_regressor_adjusted_r_squared),
            }
            pd.DataFrame(df_dict).to_csv(preamble + "desc-cvr_stats.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            # 1D data
            if "initialglobalalignedregressorseries" in data_save_list:
                # global regressor
                fname = preamble + "desc-initialGlobalAlignedRegressorSeries_timeseries"
                saveTimeseriesInfo(fname + ".json", 0, up_sampled_sample_time)
                pd.DataFrame(np.vstack((global_postproc_timeseries, initial_global_aligned_regressor_timeseries)).T, columns=["global_series", "aligned_regressor_series"]).to_csv(fname + ".tsv.gz", sep="\t", index = False, compression="gzip")
            if "globalregressorfit" in data_save_list:
                # global regressor
                fname = preamble + "desc-globalRegressorFit_timeseries"
                saveTimeseriesInfo(fname + ".json", 0, regression_sample_time)
                pd.DataFrame(np.vstack((down_sampled_global_postproc_timeseries, down_sampled_global_aligned_regressor_timeseries, down_sampled_global_regressor_predictions)).T, columns=["global_series", "aligned_regressor_series", "predictions"]).to_csv(fname + ".tsv.gz", sep="\t", index = False, compression="gzip")
            if "globalregressorxcorrelation" in data_save_list: 
                # global regressor correlations
                fname = preamble + "desc-globalRegressorrXCorrelation_timeseries"
                saveTimeseriesInfo(fname + ".json", global_regressor_timeshifts[0], up_sampled_sample_time)
                pd.Series(global_regressor_correlations).to_csv(fname + ".tsv.gz", sep="\t", index = False, header=False, compression="gzip")
            if "confounds" in data_save_list and regression_confounds_df is not None:
                # confounds
                fname = preamble + "desc-confounds_timeseries"
                saveTimeseriesInfo(fname + ".json", 0, regression_sample_time)
                regression_confounds_df.to_csv(fname + ".tsv.gz", sep="\t", index = False, compression="gzip")
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
                        "physio-tr"                             : physio_tr,
                        # global regressor data
                        "regressor-unit"                        : regressor_unit,
                        "regressor-initial-baseline"            : regressor_baseline,
                        "initial-global-regressor-alignment"    : initial_global_regressor_alignment,
                        # post-processing data  
                        "up-sampling-factor"                    : up_sampling_factor,
                        "up-sampled-sample-time"                : up_sampled_sample_time,
                        "regression-sample-time"                : regression_sample_time,
                        # physio alignment data
                        "reference-regressor-timeshift"         : reference_regressor_timeshift, 
                        "align-regressor-absolute-bounds"       : (align_regressor_absolute_lower_bound, align_regressor_absolute_upper_bound),
                        # confound names
                        "confounds-names"                       : confounds_names
                    }
            with open(preamble + "desc-data_info.json", "w") as info_file:
                json.dump(data_info, info_file, indent='\t')
            
            # save analysis dict last since this indicate analysis has been completed succesfully
            with open(analysis_file, "w") as info_file:
                json.dump(analysis_dict, info_file, indent='\t')

save_data_node = CustomNode(saveData, description="save data")
