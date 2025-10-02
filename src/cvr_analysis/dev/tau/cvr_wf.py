# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode, _and_, _or_, not_
from cvr_analysis.default.mr.p
ost_processing import post_processing_wf
from cvr_analysis.default.mr.regression import regression_wf
from cvr_analysis.default.mr.save_data import create_hash_check_override, save_data_node

# %%
##############################################
# cvr_analysis
##############################################
cvr_analysis_wf = ProcessWorkflow(
    (
        # post-processing
        (ProcessWorkflow.input._, post_processing_wf.input.all),
        # regression 
        (ProcessWorkflow.input._, regression_wf.input[('baseline_strategy', 'use_co2_regressor', 'correlation_window', 'correlation_phat', 'correlation_peak_threshold', 'correlation_multi_peak_strategy', 'align_regressor_lower_bound', 'align_regressor_upper_bound', 'nr_parallel_processes', 'show_pbar', 'maxcorr_bipolar', 'filter_timeshifts_size', 'filter_timeshifts_filter_type', 'filter_timeshifts_smooth_fwhm', 'refine_regressor_correlation_threshold', 'refine_regressor_nr_recursions', 'refine_regressor_explained_variance', 'do_dtw', 'confound_regressor_correlation_threshold')]),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_bold_timeseries, regression_wf.input.bold_timeseries),
        (post_processing_wf.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_co2_timeseries, regression_wf.input.co2_timeseries),
        (post_processing_wf.output.time_limited_temporal_filtered_detrended_up_sampled_confounds_df, regression_wf.input.confounds_df),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        (post_processing_wf.output.timeseries_masker, regression_wf.input.timeseries_masker),
        # save data
        (ProcessWorkflow.input._, save_data_node.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'space','data_to_save')]),
        # post-processing data
        (post_processing_wf.output.voxel_mask_img, save_data_node.input.voxel_mask_img),
        (post_processing_wf.output.timeseries_masker, save_data_node.input.timeseries_masker),
        (post_processing_wf.output.bold_tr, save_data_node.input.bold_tr),
        (post_processing_wf.output.up_sampling_factor, save_data_node.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, save_data_node.input.up_sampled_sample_time),
        (post_processing_wf.output.co2_event_name, save_data_node.input.co2_event_name),
        # regression data
        (regression_wf.output.down_sampled_regressor_signal_timeseries, save_data_node.input.regressor_postproc_timeseries),
        (regression_wf.output.regressor_signal_rms, save_data_node.input.regressor_rms),
        (regression_wf.output.regressor_signal_autocorrelation_timeshifts, save_data_node.input.regressor_autocorrelation_timeshifts),
        (regression_wf.output.regressor_signal_autocorrelation_correlations, save_data_node.input.regressor_autocorrelation_correlations),
        (regression_wf.output.down_sampled_regressor_aligned_co2_signal_timeseries, save_data_node.input.regressor_aligned_co2_timeseries),
        (regression_wf.output.co2_signal_rms, save_data_node.input.co2_rms),
        (regression_wf.output.co2_signal_autocorrelation_timeshifts, save_data_node.input.co2_autocorrelation_timeshifts),
        (regression_wf.output.co2_signal_autocorrelation_correlations, save_data_node.input.co2_autocorrelation_correlations),
        (regression_wf.output.regressor_co2_timeshift_maxcorr, save_data_node.input.regressor_co2_timeshift_maxcorr),
        (regression_wf.output.regressor_co2_maxcorr, save_data_node.input.regressor_co2_maxcorr),
        (regression_wf.output.regressor_co2_timeshifts, save_data_node.input.regressor_co2_timeshifts),
        (regression_wf.output.regressor_co2_correlations, save_data_node.input.regressor_co2_correlations),
        (regression_wf.output.reference_regressor_timeshift, save_data_node.input.reference_regressor_timeshift),
        (regression_wf.output.regressor_co2_beta, save_data_node.input.regressor_co2_beta),
        (regression_wf.output.align_regressor_absolute_lower_bound, save_data_node.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, save_data_node.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.boldIter_down_sampled_bold_signal_ts, save_data_node.input.bold_postproc_timeseries),
        (regression_wf.output.boldIter_timeshift_maxcorr, save_data_node.input.bold_timeshift_maxcorr),
        (regression_wf.output.boldIter_maxcorr, save_data_node.input.bold_maxcorr),
        (regression_wf.output.boldIter_timeshifts, save_data_node.input.bold_timeshifts),
        (regression_wf.output.boldIter_correlations, save_data_node.input.bold_correlations),
        (regression_wf.output.boldIter_down_sampled_aligned_regressor_signal_timeseries, save_data_node.input.bold_aligned_regressor_timeseries),
        (regression_wf.output.boldIter_dof, save_data_node.input.bold_dof),
        (regression_wf.output.boldIter_regressor_p, save_data_node.input.bold_p_value),
        (regression_wf.output.boldIter_down_sampled_bold_signal_predictions, save_data_node.input.bold_predictions),
        (regression_wf.output.boldIter_r_squared, save_data_node.input.bold_r_squared),
        (regression_wf.output.boldIter_adjusted_r_squared, save_data_node.input.bold_adjusted_r_squared),
        (regression_wf.output.boldIter_regressor_se, save_data_node.input.bold_standard_error),
        (regression_wf.output.boldIter_regressor_t, save_data_node.input.bold_t_value),
        (regression_wf.output.boldIter_regressor_beta, save_data_node.input.bold_cvr_amplitude),
        (regression_wf.output.down_sampled_sample_time, save_data_node.input.regression_down_sampled_sample_time),
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
        (ProcessWorkflow.input._, create_hash_check_override.input.all / create_hash_check_override.input.do_dtw),
        (create_hash_check_override.output.analysis_id, ProcessWorkflow.output.analysis_id),
        # conditional run
        (ProcessWorkflow.input._, conditional_run_cvr.input.all / conditional_run_cvr.input[("run_analysis", "analysis_file", "analysis_dict", "do_dtw")]),
        (create_hash_check_override.output.run_analysis, conditional_run_cvr.input.run_analysis),
        (create_hash_check_override.output.analysis_file, conditional_run_cvr.input.analysis_file),
        (create_hash_check_override.output.analysis_dict, conditional_run_cvr.input.analysis_dict),
        (conditional_run_cvr.output.all, ProcessWorkflow.output._),
        # do dtw
        (ProcessWorkflow.input.ensure_co2_units *_and_* (not_*ProcessWorkflow.input.use_co2_regressor *_or_* ProcessWorkflow.input.refine_regressor_nr_recursions > ValueNode(0).output.value), (create_hash_check_override.input.do_dtw, conditional_run_cvr.input.do_dtw)),
    ),
    "cvr wf"
).setDefaultInputs(ensure_co2_units = True)
# %%
