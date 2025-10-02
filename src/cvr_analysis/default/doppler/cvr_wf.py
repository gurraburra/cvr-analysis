# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode
from cvr_analysis.default.doppler.load_data_wf import load_data_wf
from cvr_analysis.default.helpers.workflows.post_processing_wf import post_processing_wf
from cvr_analysis.default.helpers.workflows.regression_wf import regression_wf
from cvr_analysis.default.doppler.save_data import create_hash_check_override, save_data_node

# %%
##############################################
# cvr_analysis
##############################################
cvr_analysis_wf = ProcessWorkflow(
    (
        # load data wf
        (ProcessWorkflow.input._, load_data_wf.input.all),
        # post-processing
        (ProcessWorkflow.input._, post_processing_wf.input.all - post_processing_wf.input[("confounds_df", "depvars_times", "depvars_timeseries", "depvars_tr", "global_timeseries", "regressor_times", "regressor_timeseries")]),
        (ValueNode(None).output.value, post_processing_wf.input.confounds_df),
        (load_data_wf.output.doppler_times, post_processing_wf.input.depvars_times),
        (load_data_wf.output.doppler_timeseries, post_processing_wf.input.depvars_timeseries),
        (load_data_wf.output.doppler_tr, post_processing_wf.input.depvars_tr),
        (load_data_wf.output.global_timeseries, post_processing_wf.input.global_timeseries),
        (load_data_wf.output.regressor_times, post_processing_wf.input.regressor_times),
        (load_data_wf.output.regressor_timeseries, post_processing_wf.input.regressor_timeseries),
        # regression 
        (ProcessWorkflow.input._, regression_wf.input.all - regression_wf.input[("confounds_df", "depvars_timeseries", "depvars_units", "down_sampling_factor", "global_timeseries", "psc_regressor", "regressor_timeseries", "regressor_units", "sample_time", "timeseries_masker", "filter_timeshifts_filter_type", "filter_timeshifts_size", "filter_timeshifts_smooth_fwhm", "refine_regressor_correlation_threshold", "refine_regressor_explained_variance", "refine_regressor_nr_recursions")]),
        (ProcessWorkflow.input.regressor == ValueNode("global-signal").output.value, regression_wf.input.psc_regressor),
        (ValueNode(None).output.value, regression_wf.input[("timeseries_masker", "filter_timeshifts_filter_type", "filter_timeshifts_size", "filter_timeshifts_smooth_fwhm", "refine_regressor_correlation_threshold", "refine_regressor_explained_variance")]),
        (ValueNode(0).output.value, regression_wf.input.refine_regressor_nr_recursions),
        (load_data_wf.output.doppler_units, regression_wf.input.depvars_units),
        (load_data_wf.output.regressor_units, regression_wf.input.regressor_units),
        (ValueNode(None).output.value, regression_wf.input.confounds_df),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_depvars_timeseries, regression_wf.input.depvars_timeseries),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_global_timeseries, regression_wf.input.global_timeseries),
        (post_processing_wf.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries, regression_wf.input.regressor_timeseries),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        # save data
        (ProcessWorkflow.input._, save_data_node.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'data_to_save')]),
        # post-processing data
        (load_data_wf.output.doppler_tr, save_data_node.input.doppler_tr),
        (load_data_wf.output.doppler_variables, save_data_node.input.doppler_variables),
        (post_processing_wf.output.up_sampling_factor, save_data_node.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, save_data_node.input.up_sampled_sample_time),
        (post_processing_wf.output.initial_global_regressor_alignment, save_data_node.input.initial_global_regressor_alignment),
        (post_processing_wf.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries, save_data_node.input.initial_global_aligned_regressor_timeseries),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_global_timeseries, save_data_node.input.global_postproc_timeseries),
        # regression data
        (regression_wf.output.regressor_signal_units, save_data_node.input.regressor_units),
        (regression_wf.output.depvars_signal_units, save_data_node.input.doppler_units),
        (regression_wf.output.regressor_signal_rms, save_data_node.input.regressor_rms),
        (regression_wf.output.regressor_signal_autocorrelation_timeshifts, save_data_node.input.regressor_autocorrelation_timeshifts),
        (regression_wf.output.regressor_signal_autocorrelation_correlations, save_data_node.input.regressor_autocorrelation_correlations),
        (regression_wf.output.global_signal_rms, save_data_node.input.global_rms),
        (regression_wf.output.global_signal_autocorrelation_timeshifts, save_data_node.input.global_autocorrelation_timeshifts),
        (regression_wf.output.global_signal_autocorrelation_correlations, save_data_node.input.global_autocorrelation_correlations),
        (regression_wf.output.global_regressor_timeshift_maxcorr, save_data_node.input.global_regressor_timeshift_maxcorr),
        (regression_wf.output.global_regressor_maxcorr, save_data_node.input.global_regressor_maxcorr),
        (regression_wf.output.global_regressor_timeshifts, save_data_node.input.global_regressor_timeshifts),
        (regression_wf.output.global_regressor_correlations, save_data_node.input.global_regressor_correlations),
        (regression_wf.output.reference_regressor_timeshift, save_data_node.input.reference_regressor_timeshift),
        (regression_wf.output.global_regressor_beta, save_data_node.input.global_regressor_beta),
        (regression_wf.output.align_regressor_absolute_lower_bound, save_data_node.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, save_data_node.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.depvarsIter_down_sampled_depvars_signal_timeseries, save_data_node.input.doppler_postproc_timeseries),
        (regression_wf.output.depvarsIter_timeshift_maxcorr, save_data_node.input.doppler_timeshift_maxcorr),
        (regression_wf.output.depvarsIter_maxcorr, save_data_node.input.doppler_maxcorr),
        (regression_wf.output.depvarsIter_timeshifts, save_data_node.input.doppler_timeshifts),
        (regression_wf.output.depvarsIter_correlations, save_data_node.input.doppler_correlations),
        (regression_wf.output.depvarsIter_down_sampled_aligned_regressor_signal_timeseries, save_data_node.input.doppler_aligned_regressor_timeseries),
        (regression_wf.output.depvarsIter_dof, save_data_node.input.doppler_dof),
        (regression_wf.output.depvarsIter_regressor_p, save_data_node.input.doppler_p_value),
        (regression_wf.output.depvarsIter_down_sampled_depvars_signal_predictions, save_data_node.input.doppler_predictions),
        (regression_wf.output.depvarsIter_r_squared, save_data_node.input.doppler_r_squared),
        (regression_wf.output.depvarsIter_adjusted_r_squared, save_data_node.input.doppler_adjusted_r_squared),
        (regression_wf.output.depvarsIter_regressor_se, save_data_node.input.doppler_standard_error),
        (regression_wf.output.depvarsIter_regressor_t, save_data_node.input.doppler_t_value),
        (regression_wf.output.depvarsIter_regressor_beta, save_data_node.input.doppler_cvr_amplitude),
        (regression_wf.output.down_sampled_sample_time, save_data_node.input.regression_sample_time),
        (regression_wf.output.down_sampled_global_signal_timeseries, save_data_node.input.global_signal_timeseries),
        (regression_wf.output.down_sampled_global_aligned_regressor_signal_timeseries, save_data_node.input.global_aligned_regressor_timeseries),
        (regression_wf.output.global_regressor_predictions, save_data_node.input.global_regressor_predictions),
        # map save data output
        (save_data_node.output.output, ProcessWorkflow.output.cvr_analysis_dummy_output),
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

