# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode, _and_
from cvr_analysis.default.mr.load_data_wf import load_data_wf
from cvr_analysis.default.helpers.workflows.post_processing_wf import post_processing_wf
from cvr_analysis.default.helpers.workflows.regression_wf import regression_wf
from cvr_analysis.default.helpers.workflows.get_confounds import get_regression_confounds_wf
from cvr_analysis.default.helpers.workflows.refine_regressor import refine_regressor_wf
from cvr_analysis.default.mr.save_data import create_hash_check_override, save_data_node

# %%
##############################################
# cvr_analysis
##############################################
cvr_analysis_wf = ProcessWorkflow(
    (
        # load data wf
        (ProcessWorkflow.input._, load_data_wf.input.all),
        # post-processing
        (ProcessWorkflow.input._, post_processing_wf.input.all - post_processing_wf.input[("confounds_df", "depvars_times", "depvars_timeseries", "depvars_unit", "depvars_tr", "global_timeseries", "regressor_times", "regressor_timeseries", "regressor_unit", "psc_regressor", "psc_depvars")]),
        (ValueNode(True).output.value, post_processing_wf.input.psc_depvars),
        (ProcessWorkflow.input.regressor == ValueNode("global-signal").output.value, post_processing_wf.input.psc_regressor),
        (load_data_wf.output.confounds_df, post_processing_wf.input.confounds_df),
        (load_data_wf.output.bold_times, post_processing_wf.input.depvars_times),
        (load_data_wf.output.bold_timeseries, post_processing_wf.input.depvars_timeseries),
        (load_data_wf.output.bold_tr, post_processing_wf.input.depvars_tr),
        (load_data_wf.output.global_timeseries, post_processing_wf.input.global_timeseries),
        (load_data_wf.output.regressor_times, post_processing_wf.input.regressor_times),
        (load_data_wf.output.regressor_timeseries, post_processing_wf.input.regressor_timeseries),
        (load_data_wf.output.bold_unit, post_processing_wf.input.depvars_unit),
        (load_data_wf.output.regressor_unit, post_processing_wf.input.regressor_unit),
        # get confounds
        (ProcessWorkflow.input._, get_regression_confounds_wf.input[("drift_high_pass", "drift_model", "drift_order", "include_drift_confounds", "include_motion_confounds", "include_spike_confounds", "motion_derivatives", "motion_powers", "spike_diff_cutoff", "spike_global_cutoff")]),
        (post_processing_wf.output.confounds_postproc_df, get_regression_confounds_wf.input.confounds_df),
        (post_processing_wf.output.depvars_postproc_timeseries, get_regression_confounds_wf.input.depvars_timeseries),
        (post_processing_wf.output.global_postproc_timeseries, get_regression_confounds_wf.input.global_timeseries),
        (post_processing_wf.output.up_sampling_factor, get_regression_confounds_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, get_regression_confounds_wf.input.time_step),
        # refine regressor wf
        (ProcessWorkflow.input._, refine_regressor_wf.input.all / refine_regressor_wf.input[("depvars_timeseries", "regressor_timeseries", "sample_time")]),
        (post_processing_wf.output.depvars_postproc_timeseries, refine_regressor_wf.input.depvars_timeseries),
        (post_processing_wf.output.regressor_postproc_timeseries, refine_regressor_wf.input.regressor_timeseries),
        (post_processing_wf.output.up_sampled_sample_time, refine_regressor_wf.input.sample_time),
        # regression 
        (ProcessWorkflow.input._, regression_wf.input.all - regression_wf.input[("depvars_timeseries", "down_sampling_factor", "global_timeseries", "regressor_timeseries", "sample_time", "timeseries_masker", "down_sampled_regression_confounds_df", "reference_regressor_timeshift")]),
        (load_data_wf.output.timeseries_masker, regression_wf.input.timeseries_masker),
        (post_processing_wf.output.depvars_postproc_timeseries, regression_wf.input.depvars_timeseries),
        (post_processing_wf.output.global_postproc_timeseries, regression_wf.input.global_timeseries),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_df, regression_wf.input.down_sampled_regression_confounds_df),
        (refine_regressor_wf.output.refined_regressor_timeseries, regression_wf.input.regressor_timeseries),
        (refine_regressor_wf.output.reference_regressor_timeshift, regression_wf.input.reference_regressor_timeshift),
        # save data
        (ProcessWorkflow.input._, save_data_node.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'space', 'data_to_save')]),
        # post-processing data
        (load_data_wf.output.bold_tr, save_data_node.input.bold_tr),
        (load_data_wf.output.timeseries_masker, save_data_node.input.timeseries_masker),
        (load_data_wf.output.voxel_mask_img, save_data_node.input.voxel_mask_img),
        (post_processing_wf.output.up_sampling_factor, save_data_node.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, save_data_node.input.up_sampled_sample_time),
        (post_processing_wf.output.initial_global_regressor_alignment, save_data_node.input.initial_global_regressor_alignment),
        (post_processing_wf.output.regressor_postproc_timeseries, save_data_node.input.initial_global_aligned_regressor_timeseries),
        (post_processing_wf.output.global_postproc_timeseries, save_data_node.input.global_postproc_timeseries),
        (post_processing_wf.output.regressor_postproc_unit, save_data_node.input.regressor_unit),
        (post_processing_wf.output.depvars_postproc_unit, save_data_node.input.bold_unit),
        (post_processing_wf.output.depvars_postproc_baseline, save_data_node.input.bold_baseline),
        (post_processing_wf.output.regressor_postproc_baseline, save_data_node.input.regressor_baseline),
        (post_processing_wf.output.global_postproc_baseline, save_data_node.input.global_baseline),
        (post_processing_wf.output.up_sampled_sample_time / post_processing_wf.output.up_sampling_factor, save_data_node.input.regression_sample_time),
        # get confounds
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_df, save_data_node.input.regression_confounds_df),
        # refine regressor
        (regression_wf.output.reference_regressor_timeshift, save_data_node.input.reference_regressor_timeshift),
        # regression data
        (regression_wf.output.global_regressor_timeshift_maxcorr, save_data_node.input.global_regressor_timeshift_maxcorr),
        (regression_wf.output.global_regressor_maxcorr, save_data_node.input.global_regressor_maxcorr),
        (regression_wf.output.global_regressor_timeshifts, save_data_node.input.global_regressor_timeshifts),
        (regression_wf.output.global_regressor_correlations, save_data_node.input.global_regressor_correlations),
        (regression_wf.output.global_regressor_beta, save_data_node.input.global_regressor_beta),
        (regression_wf.output.down_sampled_global_timeseries, save_data_node.input.down_sampled_global_postproc_timeseries),
        (regression_wf.output.down_sampled_global_aligned_regressor_timeseries, save_data_node.input.down_sampled_global_aligned_regressor_timeseries),
        (regression_wf.output.down_sampled_global_regressor_predictions, save_data_node.input.down_sampled_global_regressor_predictions),
        (regression_wf.output.align_regressor_absolute_lower_bound, save_data_node.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, save_data_node.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.depvarsIter_down_sampled_depvars_timeseries, save_data_node.input.bold_postproc_timeseries),
        (regression_wf.output.depvarsIter_timeshift_maxcorr, save_data_node.input.bold_timeshift_maxcorr),
        (regression_wf.output.depvarsIter_maxcorr, save_data_node.input.bold_maxcorr),
        (regression_wf.output.depvarsIter_timeshifts, save_data_node.input.bold_timeshifts),
        (regression_wf.output.depvarsIter_correlations, save_data_node.input.bold_correlations),
        (regression_wf.output.depvarsIter_down_sampled_aligned_regressor_timeseries, save_data_node.input.bold_aligned_regressor_timeseries),
        (regression_wf.output.depvarsIter_dof, save_data_node.input.bold_dof),
        (regression_wf.output.depvarsIter_regressor_p, save_data_node.input.bold_p_value),
        (regression_wf.output.depvarsIter_down_sampled_depvars_predictions, save_data_node.input.bold_predictions),
        (regression_wf.output.depvarsIter_r_squared, save_data_node.input.bold_r_squared),
        (regression_wf.output.depvarsIter_adjusted_r_squared, save_data_node.input.bold_adjusted_r_squared),
        (regression_wf.output.depvarsIter_regressor_se, save_data_node.input.bold_standard_error),
        (regression_wf.output.depvarsIter_regressor_t, save_data_node.input.bold_t_value),
        (regression_wf.output.depvarsIter_regressor_beta, save_data_node.input.bold_cvr_amplitude),
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
        # 
        (ProcessWorkflow.input.regressor, ProcessWorkflow.output._),
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
)#.setDefaultInputs(ensure_regressor_unit = True, include_motion_confounds = False)
# %%