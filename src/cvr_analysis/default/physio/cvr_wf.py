# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode, _and_
from process_control import ProcessWorkflow, ConditionalNode, ValueNode
from cvr_analysis.default.physio.load_data_wf import load_data_wf
from cvr_analysis.default.helpers.workflows.post_processing_wf import post_processing_wf
from cvr_analysis.default.helpers.workflows.regression_wf import regression_wf
from cvr_analysis.default.physio.save_data import create_hash_check_override, save_data_node

# %%
##############################################
# cvr_analysis
##############################################
cvr_analysis_wf = ProcessWorkflow(
    (
        # load data wf
        (ProcessWorkflow.input._, load_data_wf.input.all),
        # post-processing
        (ProcessWorkflow.input._, post_processing_wf.input.all - post_processing_wf.input[("confounds_df", "depvars_times", "depvars_timeseries", "depvars_unit", "depvars_tr", "psc_depvars", "global_timeseries", "regressor_times", "regressor_timeseries", "regressor_unit")]),
        (ProcessWorkflow.input.psc_physio, post_processing_wf.input.psc_depvars),
        (ValueNode(None).output.value, post_processing_wf.input.confounds_df),
        (load_data_wf.output.physio_times, post_processing_wf.input.depvars_times),
        (load_data_wf.output.physio_timeseries, post_processing_wf.input.depvars_timeseries),
        (load_data_wf.output.physio_unit, post_processing_wf.input.depvars_unit),
        (load_data_wf.output.physio_tr, post_processing_wf.input.depvars_tr),
        (load_data_wf.output.global_timeseries, post_processing_wf.input.global_timeseries),
        (load_data_wf.output.regressor_times, post_processing_wf.input.regressor_times),
        (load_data_wf.output.regressor_timeseries, post_processing_wf.input.regressor_timeseries),
        (load_data_wf.output.regressor_unit, post_processing_wf.input.regressor_unit),
        # regression 
        (ProcessWorkflow.input._, regression_wf.input.all - regression_wf.input[("confounds_postproc_df", "depvars_postproc_timeseries", "down_sampling_factor", "global_postproc_timeseries", "regressor_postproc_timeseries", "sample_time", "timeseries_masker", "filter_timeshifts_filter_type", "filter_timeshifts_size", "filter_timeshifts_smooth_fwhm", "include_motion_confounds", "confound_regressor_correlation_threshold", "motion_derivatives", "motion_powers")]),
        (ValueNode(None).output.value, regression_wf.input[("timeseries_masker", "filter_timeshifts_filter_type", "filter_timeshifts_size", "filter_timeshifts_smooth_fwhm", "confound_regressor_correlation_threshold", "confounds_postproc_df")]),
        (ValueNode(False).output.value, regression_wf.input[("include_motion_confounds", "motion_derivatives", "motion_powers")]),
        (post_processing_wf.output.depvars_postproc_timeseries, regression_wf.input.depvars_postproc_timeseries),
        (post_processing_wf.output.global_postproc_timeseries, regression_wf.input.global_postproc_timeseries),
        (post_processing_wf.output.regressor_postproc_timeseries, regression_wf.input.regressor_postproc_timeseries),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        (post_processing_wf.output.regressor_postproc_unit, save_data_node.input.regressor_unit),
        (post_processing_wf.output.depvars_postproc_unit, save_data_node.input.physio_unit),
        (post_processing_wf.output.depvars_postproc_baseline, save_data_node.input.physio_baseline),
        (post_processing_wf.output.regressor_postproc_baseline, save_data_node.input.regressor_baseline),
        (post_processing_wf.output.global_postproc_baseline, save_data_node.input.global_baseline),
        # save data
        (ProcessWorkflow.input._, save_data_node.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'data_to_save')]),
        # post-processing data
        (load_data_wf.output.physio_tr, save_data_node.input.physio_tr),
        (load_data_wf.output.physio_variables, save_data_node.input.physio_variables),
        (post_processing_wf.output.up_sampling_factor, save_data_node.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, save_data_node.input.up_sampled_sample_time),
        (post_processing_wf.output.initial_global_regressor_alignment, save_data_node.input.initial_global_regressor_alignment),
        (post_processing_wf.output.regressor_postproc_timeseries, save_data_node.input.initial_global_aligned_regressor_timeseries),
        (post_processing_wf.output.global_postproc_timeseries, save_data_node.input.global_postproc_timeseries),
        # regression data
        (regression_wf.output.down_sampled_regression_confounds_postproc_df, save_data_node.input.regression_confounds_df),
        (regression_wf.output.regressor_postproc_rms, save_data_node.input.regressor_rms),
        (regression_wf.output.regressor_postproc_autocorrelation_timeshifts, save_data_node.input.regressor_autocorrelation_timeshifts),
        (regression_wf.output.regressor_postproc_autocorrelation_correlations, save_data_node.input.regressor_autocorrelation_correlations),
        (regression_wf.output.global_postproc_rms, save_data_node.input.global_rms),
        (regression_wf.output.global_postproc_autocorrelation_timeshifts, save_data_node.input.global_autocorrelation_timeshifts),
        (regression_wf.output.global_postproc_autocorrelation_correlations, save_data_node.input.global_autocorrelation_correlations),
        (regression_wf.output.global_regressor_timeshift_maxcorr, save_data_node.input.global_regressor_timeshift_maxcorr),
        (regression_wf.output.global_regressor_maxcorr, save_data_node.input.global_regressor_maxcorr),
        (regression_wf.output.global_regressor_timeshifts, save_data_node.input.global_regressor_timeshifts),
        (regression_wf.output.global_regressor_correlations, save_data_node.input.global_regressor_correlations),
        (regression_wf.output.global_regressor_beta, save_data_node.input.global_regressor_cvr_amplitude),
        (regression_wf.output.global_regressor_p_value, save_data_node.input.global_regressor_p_value),
        (regression_wf.output.global_regressor_se, save_data_node.input.global_regressor_standard_error),
        (regression_wf.output.global_regressor_t_value, save_data_node.input.global_regressor_t_value),
        (regression_wf.output.global_regressor_dof, save_data_node.input.global_regressor_dof),
        (regression_wf.output.global_regressor_r_squared, save_data_node.input.global_regressor_r_squared),
        (regression_wf.output.global_regressor_adjusted_r_squared, save_data_node.input.global_regressor_adjusted_r_squared),
        (regression_wf.output.reference_regressor_timeshift, save_data_node.input.reference_regressor_timeshift),
        (regression_wf.output.align_regressor_absolute_lower_bound, save_data_node.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, save_data_node.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.depvarsIter_down_sampled_depvars_postproc_timeseries, save_data_node.input.physio_postproc_timeseries),
        (regression_wf.output.depvarsIter_timeshift_maxcorr, save_data_node.input.physio_timeshift_maxcorr),
        (regression_wf.output.depvarsIter_maxcorr, save_data_node.input.physio_maxcorr),
        (regression_wf.output.depvarsIter_timeshifts, save_data_node.input.physio_timeshifts),
        (regression_wf.output.depvarsIter_correlations, save_data_node.input.physio_correlations),
        (regression_wf.output.depvarsIter_down_sampled_aligned_regressor_postproc_timeseries, save_data_node.input.physio_aligned_regressor_timeseries),
        (regression_wf.output.depvarsIter_dof, save_data_node.input.physio_dof),
        (regression_wf.output.depvarsIter_regressor_p, save_data_node.input.physio_p_value),
        (regression_wf.output.depvarsIter_down_sampled_depvars_postproc_predictions, save_data_node.input.physio_predictions),
        (regression_wf.output.depvarsIter_r_squared, save_data_node.input.physio_r_squared),
        (regression_wf.output.depvarsIter_adjusted_r_squared, save_data_node.input.physio_adjusted_r_squared),
        (regression_wf.output.depvarsIter_regressor_se, save_data_node.input.physio_standard_error),
        (regression_wf.output.depvarsIter_regressor_t, save_data_node.input.physio_t_value),
        (regression_wf.output.depvarsIter_regressor_beta, save_data_node.input.physio_cvr_amplitude),
        (regression_wf.output.down_sampled_sample_time, save_data_node.input.regression_sample_time),
        (regression_wf.output.down_sampled_global_postproc_timeseries, save_data_node.input.down_sampled_global_postproc_timeseries),
        (regression_wf.output.down_sampled_global_aligned_regressor_postproc_timeseries, save_data_node.input.down_sampled_global_aligned_regressor_timeseries),
        (regression_wf.output.down_sampled_global_regressor_predictions, save_data_node.input.down_sampled_global_regressor_predictions),
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
        (ProcessWorkflow.input.ensure_regressor_unit *_and_* (ProcessWorkflow.input.refine_regressor_nr_recursions > ValueNode(0).output.value), (create_hash_check_override.input.do_dtw, conditional_run_cvr.input.do_dtw)),
    ),
    "cvr wf"
)
# %%

