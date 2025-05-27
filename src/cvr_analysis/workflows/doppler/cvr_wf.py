# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode, _and_, _or_, not_
from cvr_analysis.workflows.doppler.post_processing import post_processing_wf
from cvr_analysis.workflows.doppler.regression import regression_wf
from cvr_analysis.workflows.doppler.save_data import create_hash_check_override, conditionally_save_data

# %%
##############################################
# cvr_analysis
##############################################
cvr_analysis_wf = ProcessWorkflow(
    (
        # post-processing
        (ProcessWorkflow.input._, post_processing_wf.input.all),
        # regression 
        (ProcessWorkflow.input._, regression_wf.input[('baseline_strategy', 'correlation_window', 'correlation_phat', 'correlation_peak_threshold', 'correlation_multi_peak_strategy', 'align_regressor_lower_bound', 'align_regressor_upper_bound', 'global_align_regressor_lower_bound', 'global_align_regressor_upper_bound', 'show_pbar', 'maxcorr_bipolar')]),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries, regression_wf.input.doppler_timeseries),
        (post_processing_wf.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries, regression_wf.input.regressor_timeseries),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        # save data
        (ProcessWorkflow.input._, conditionally_save_data.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'save_data','full_output')]),
        # post-processing data
        (post_processing_wf.output.doppler_tr, conditionally_save_data.input.doppler_tr),
        (post_processing_wf.output.up_sampling_factor, conditionally_save_data.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, conditionally_save_data.input.up_sampled_sample_time),
        (post_processing_wf.output.regressor_event_name, conditionally_save_data.input.regressor_event_name),
        (post_processing_wf.output.doppler_headers, conditionally_save_data.input.doppler_headers),
        (post_processing_wf.output.doppler_units, conditionally_save_data.input.doppler_units),
        (post_processing_wf.output.initial_global_regressor_timeshift, conditionally_save_data.input.initial_global_regressor_timeshift),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries.mean(axis = 0), conditionally_save_data.input.doppler_means),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries.max(axis = 0), conditionally_save_data.input.doppler_maxs),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries.min(axis = 0), conditionally_save_data.input.doppler_mins),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries.std(axis = 0), conditionally_save_data.input.doppler_stds),
        # regression data
        (regression_wf.output.regressor_signal_rms, conditionally_save_data.input.regressor_rms),
        (regression_wf.output.regressor_signal_autocorrelation_timeshifts, conditionally_save_data.input.regressor_autocorrelation_timeshifts),
        (regression_wf.output.regressor_signal_autocorrelation_correlations, conditionally_save_data.input.regressor_autocorrelation_correlations),
        (regression_wf.output.refined_global_regressor_timeshift, conditionally_save_data.input.refined_global_regressor_timeshift),
        (regression_wf.output.align_regressor_absolute_lower_bound, conditionally_save_data.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, conditionally_save_data.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.dopplerIter_down_sampled_doppler_signal_ts, conditionally_save_data.input.doppler_postproc_timeseries),
        (regression_wf.output.dopplerIter_timeshift_maxcorr, conditionally_save_data.input.doppler_timeshift_maxcorr),
        (regression_wf.output.dopplerIter_maxcorr, conditionally_save_data.input.doppler_maxcorr),
        (regression_wf.output.dopplerIter_timeshifts, conditionally_save_data.input.doppler_timeshifts),
        (regression_wf.output.dopplerIter_correlations, conditionally_save_data.input.doppler_correlations),
        (regression_wf.output.dopplerIter_down_sampled_aligned_regressor_signal_timeseries, conditionally_save_data.input.doppler_aligned_regressor_timeseries),
        (regression_wf.output.dopplerIter_dof, conditionally_save_data.input.doppler_dof),
        (regression_wf.output.dopplerIter_down_sampled_doppler_signal_predictions, conditionally_save_data.input.doppler_predictions),
        (regression_wf.output.dopplerIter_r_squared, conditionally_save_data.input.doppler_r_squared),
        (regression_wf.output.dopplerIter_adjusted_r_squared, conditionally_save_data.input.doppler_adjusted_r_squared),
        (regression_wf.output.dopplerIter_regressor_p, conditionally_save_data.input.doppler_p_value),
        (regression_wf.output.dopplerIter_regressor_se, conditionally_save_data.input.doppler_se),
        (regression_wf.output.dopplerIter_regressor_beta, conditionally_save_data.input.doppler_cvr_amplitude),
        (regression_wf.output.down_sampled_sample_time, conditionally_save_data.input.regression_down_sampled_sample_time),
        # map save data output
        (conditionally_save_data.output.output, ProcessWorkflow.output.cvr_analysis_dummy_output),
    ),
    description="cvr analysis wf"
)

# %%

##############################################
# conditionally run cvr analysis
##############################################

conditional_run_cvr = ConditionalNode("run_analysis", {True : cvr_analysis_wf, False : None}, description="Conditionally run CVR analysis")

# cvr wf including the conditional node to check if analysis already performed
cvr_wf = ProcessWorkflow(
    (
        # create hash check override
        (ProcessWorkflow.input._, create_hash_check_override.input.all),
        (create_hash_check_override.output.analysis_id, ProcessWorkflow.output.analysis_id),
        # conditional run
        (ProcessWorkflow.input._, conditional_run_cvr.input.all / conditional_run_cvr.input[("run_analysis", "analysis_file", "analysis_dict")]),
        (create_hash_check_override.output.run_analysis, conditional_run_cvr.input.run_analysis),
        (create_hash_check_override.output.analysis_file, conditional_run_cvr.input.analysis_file),
        (create_hash_check_override.output.analysis_dict, conditional_run_cvr.input.analysis_dict),
        (conditional_run_cvr.output.all, ProcessWorkflow.output._),
    ),
    "cvr wf"
)
# %%
