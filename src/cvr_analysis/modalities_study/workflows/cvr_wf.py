# %%
from process_control import ProcessWorkflow, ConditionalNode, ValueNode, _and_, _or_, not_
from cvr_analysis.modalities_study.workflows.post_processing import post_processing_wf
from cvr_analysis.modalities_study.workflows.regression import regression_wf
from cvr_analysis.modalities_study.workflows.save_data import create_hash_check_override, conditionally_save_data

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
        (post_processing_wf.output.temporal_filtered_detrended_time_limited_up_sampled_bold_timeseries, regression_wf.input.bold_timeseries),
        (post_processing_wf.output.temporal_filtered_detrended_time_limited_global_aligned_up_sampled_co2_timeseries, regression_wf.input.co2_timeseries),
        (post_processing_wf.output.temporal_filtered_detrended_time_limited_up_sampled_confounds_df, regression_wf.input.confounds_df),
        (post_processing_wf.output.up_sampling_factor, regression_wf.input.down_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, regression_wf.input.sample_time),
        (post_processing_wf.output.timeseries_masker, regression_wf.input.timeseries_masker),
        # save data
        (ProcessWorkflow.input._, conditionally_save_data.input[('analysis_file', 'analysis_dict', 'subject', 'session', 'task', 'run', 'space','save_data','full_output')]),
        # post-processing data
        (post_processing_wf.output.voxel_mask_img, conditionally_save_data.input.voxel_mask_img),
        (post_processing_wf.output.timeseries_masker, conditionally_save_data.input.timeseries_masker),
        (post_processing_wf.output.bold_tr, conditionally_save_data.input.bold_tr),
        (post_processing_wf.output.up_sampling_factor, conditionally_save_data.input.up_sampling_factor),
        (post_processing_wf.output.up_sampled_sample_time, conditionally_save_data.input.up_sampled_sample_time),
        (post_processing_wf.output.co2_event_name, conditionally_save_data.input.co2_event_name),
        # regression data
        (regression_wf.output.down_sampled_regressor_signal_timeseries, conditionally_save_data.input.regressor_preproc_timeseries),
        (regression_wf.output.regressor_signal_rms, conditionally_save_data.input.regressor_rms),
        (regression_wf.output.regressor_signal_autocorrelation_timeshifts, conditionally_save_data.input.regressor_autocorrelation_timeshifts),
        (regression_wf.output.regressor_signal_autocorrelation_correlations, conditionally_save_data.input.regressor_autocorrelation_correlations),
        (regression_wf.output.down_sampled_regressor_aligned_co2_signal_timeseries, conditionally_save_data.input.regressor_aligned_co2_timeseries),
        (regression_wf.output.co2_signal_rms, conditionally_save_data.input.co2_rms),
        (regression_wf.output.co2_signal_autocorrelation_timeshifts, conditionally_save_data.input.co2_autocorrelation_timeshifts),
        (regression_wf.output.co2_signal_autocorrelation_correlations, conditionally_save_data.input.co2_autocorrelation_correlations),
        (regression_wf.output.regressor_co2_timeshift_maxcorr, conditionally_save_data.input.regressor_co2_timeshift_maxcorr),
        (regression_wf.output.regressor_co2_maxcorr, conditionally_save_data.input.regressor_co2_maxcorr),
        (regression_wf.output.regressor_co2_timeshifts, conditionally_save_data.input.regressor_co2_timeshifts),
        (regression_wf.output.regressor_co2_correlations, conditionally_save_data.input.regressor_co2_correlations),
        (regression_wf.output.reference_regressor_timeshift, conditionally_save_data.input.reference_regressor_timeshift),
        (regression_wf.output.regressor_co2_beta, conditionally_save_data.input.regressor_co2_beta),
        (regression_wf.output.align_regressor_absolute_lower_bound, conditionally_save_data.input.align_regressor_absolute_lower_bound),
        (regression_wf.output.align_regressor_absolute_upper_bound, conditionally_save_data.input.align_regressor_absolute_upper_bound),
        (regression_wf.output.boldIter_down_sampled_bold_signal_ts, conditionally_save_data.input.bold_preproc_timeseries),
        (regression_wf.output.boldIter_timeshift_maxcorr, conditionally_save_data.input.bold_timeshift_maxcorr),
        (regression_wf.output.boldIter_maxcorr, conditionally_save_data.input.bold_maxcorr),
        (regression_wf.output.boldIter_timeshifts, conditionally_save_data.input.bold_timeshifts),
        (regression_wf.output.boldIter_correlations, conditionally_save_data.input.bold_correlations),
        (regression_wf.output.boldIter_down_sampled_aligned_regressor_signal_timeseries, conditionally_save_data.input.bold_aligned_regressor_timeseries),
        (regression_wf.output.boldIter_dof, conditionally_save_data.input.bold_dof),
        (regression_wf.output.boldIter_nr_predictors, conditionally_save_data.input.bold_nr_predictors),
        (regression_wf.output.boldIter_down_sampled_bold_signal_predictions, conditionally_save_data.input.bold_predictions),
        (regression_wf.output.boldIter_r_squared, conditionally_save_data.input.bold_r_squared),
        (regression_wf.output.boldIter_adjusted_r_squared, conditionally_save_data.input.bold_adjusted_r_squared),
        (regression_wf.output.boldIter_tsnr, conditionally_save_data.input.bold_tsnr),
        (regression_wf.output.boldIter_regressor_beta, conditionally_save_data.input.bold_cvr_amplitude),
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
