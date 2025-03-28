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
                        output_directory, subject, session, task, run,
                                analysis_start_time, analysis_end_time, min_sample_freq, add_global_signal,
                                    detrend_linear_order, temporal_filter_freq, baseline_strategy, 
                                            global_align_regressor_lower_bound, global_align_regressor_upper_bound,
                                                maxcorr_bipolar, align_regressor_lower_bound, align_regressor_upper_bound, 
                                                    correlation_phat, correlation_window, correlation_multi_peak_strategy, correlation_peak_threshold,
                                                        force_run = False): 
    # folder for files
    analysis_name = "TCD-CVR-version-" + __version__ 
    files_folder = os.path.join(output_directory, f"sub-{subject}", f"ses-{session}", analysis_name)

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

    # check if refining regressor, if not change parameters
    if correlation_multi_peak_strategy is None:
        correlation_peak_threshold = None

    # check if aligning, if not change parameters
    if align_regressor_lower_bound is not None and align_regressor_upper_bound is not None and align_regressor_lower_bound == align_regressor_upper_bound:
        maxcorr_bipolar = None
   
    # analysis info
    analysis_info = {
        "analysis-name"                             : str(analysis_name),
        "min-sample-freq"                           : try_conv(min_sample_freq, float),
        "analysis-bounds"                           : try_conv((analysis_start_time, analysis_end_time), float),
        "add-global-signal"                         : try_conv(add_global_signal, bool),
        "detrend-linear-order"                      : try_conv(detrend_linear_order, int),
        "temporal-filter-freq"                      : try_conv(temporal_filter_freq, float),
        "baseline-strategy"                         : try_conv(baseline_strategy, str),
        "global-align-regressor-bounds"             : try_conv((global_align_regressor_lower_bound, global_align_regressor_upper_bound), float),
        "align-regressor-bounds"                    : try_conv((align_regressor_lower_bound, align_regressor_upper_bound), float),
        "maxcorr-bipolar"                           : try_conv(maxcorr_bipolar, bool),
        "correlation-phat"                          : bool(correlation_phat),
        "correlation-window"                        : try_conv(correlation_window, str),
        "correlation-multi-peak-strategy"           : try_conv(correlation_multi_peak_strategy, str),
        "correlation-peak-threshold"                : try_conv(correlation_peak_threshold, float),
    }

    # analysis id
    analysis_id = hashlib.sha1(str(tuple(analysis_info.items())).encode("UTF-8")).hexdigest()[:15]
    # create preamble
    def getBStr(var, val):
        if val is not None:
            return f"{var}-{val}_"
        else:
            return ""
    analysis_file = os.path.join(files_folder, f'sub-{subject}_{getBStr("ses",session)}{getBStr("task",task)}{getBStr("run",run)}analys-{analysis_id}_desc-analys_info.json')
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
                subject, session, task, run,
                    # post-processing data
                    doppler_tr, regressor_event_name,
                        doppler_headers, doppler_units, doppler_means, doppler_maxs, doppler_mins, doppler_stds,
                            up_sampling_factor, up_sampled_sample_time,
                                # regressor data
                                    regressor_rms, regressor_autocorrelation_timeshifts, regressor_autocorrelation_correlations, 
                                        # doppler alignement data
                                            doppler_postproc_timeseries, doppler_timeshift_maxcorr, doppler_maxcorr, doppler_timeshifts, doppler_correlations,
                                                doppler_aligned_regressor_timeseries,
                                                    # doppler regression data
                                                    doppler_dof, doppler_predictions, doppler_r_squared, doppler_adjusted_r_squared, doppler_tsnr, 
                                                        doppler_cvr_amplitude, doppler_p_value, regression_down_sampled_sample_time,
                                                            # full output
                                                            full_output = False) -> tuple:
        
        # make output folder if not exist
        path_folder = Path(os.path.split(analysis_file)[0])
        path_folder.mkdir(parents=True, exist_ok=True)
        # get preable
        preamble = analysis_file.split("desc-")[0]
        # 2D data
        if full_output:
            pd.DataFrame(doppler_postproc_timeseries.T, columns=doppler_headers).to_csv(preamble + "desc-postproc_doppler.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            pd.DataFrame(doppler_predictions.T, columns=doppler_headers).to_csv(preamble + "desc-predictions_doppler.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            pd.DataFrame(doppler_correlations.T, columns=doppler_headers).to_csv(preamble + "desc-correlations_timeseries.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
            pd.DataFrame(doppler_aligned_regressor_timeseries.T, columns=doppler_headers).to_csv(preamble + "desc-alignedRegressor_timeseries.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
        # dataframe data
        df_dict = {
            "signal" : doppler_headers,
            "unit" : doppler_units,
            "mean" : doppler_means,
            "max" : doppler_maxs,
            "min" : doppler_mins,
            "std" : doppler_stds,
            "cvr" : doppler_cvr_amplitude,
            "timeshift" : doppler_timeshift_maxcorr,
            "maxcorr" : doppler_maxcorr,
            "pvalue" : doppler_p_value,
            "dof" : doppler_dof,
            "rsquared" : doppler_r_squared,
            "adj_rsquared" : doppler_adjusted_r_squared,
        }
        pd.DataFrame(df_dict).to_csv(preamble + "desc-cvr_stats.tsv.gz", sep="\t", index = False, header = True, compression="gzip")
         # 1D data
        if full_output:
            def saveTimeseriesInfo(name, start_time, time_step):
                dict_ = {"start-time" : start_time, "time-step" : time_step}
                with open(name, "w") as file:
                    json.dump(dict_, file, indent='\t')
            # regressor autocorrelation
            saveTimeseriesInfo(preamble + "desc-regressorAutocorrelations_timeseries.json", regressor_autocorrelation_timeshifts[0], up_sampled_sample_time)
            pd.Series(regressor_autocorrelation_correlations).to_csv(preamble + "desc-regressorAutocorrelations_timeseries.tsv.gz", sep="\t", index = False, header = False, compression="gzip")
        
        # 1D data
        data_info = {
                    # pre-processing data
                    "subject"                           : subject,
                    "session"                           : session,
                    "task"                              : task,
                    "run"                               : run, 
                    "doppler-tr"                        : doppler_tr,
                    # post-processing data
                    "up-sampling-factor"                : up_sampling_factor,
                    "up-sampled-sample-time"            : up_sampled_sample_time,
                    "regressor-name"                    : regressor_event_name,
                    # regressor data
                    "regressor-rms"                     : regressor_rms,
                    # doppler alignment data
                    "align-regressor-time-step"         : up_sampled_sample_time,
                    "align-regressor-start-time"        : doppler_timeshifts[0,0],
                }
        with open(preamble + "desc-data_info.json", "w") as info_file:
            json.dump(data_info, info_file, indent='\t')
        
        # save analysis dict last since this indicate analysis has been completed succesfully
        with open(analysis_file, "w") as info_file:
            json.dump(analysis_dict, info_file, indent='\t')

save_data_node = CustomNode(saveData, description="save data")

conditionally_save_data = ConditionalNode(conditional_input = "save_data", default_condition = True, condition_node_map = {True : save_data_node, False : None}, description = "Save data conditionally")

