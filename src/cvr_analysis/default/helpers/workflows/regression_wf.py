# %%
# process control
from process_control import *
from cvr_analysis.default.helpers.classes.signal_processing import DownsampleTimeSeries
from cvr_analysis.default.helpers.classes.data_computation import Correlate, AlignTimeSeries, RegressCVR
from cvr_analysis.default.helpers.workflows.refine_regressor import find_timeshift_wf

__all__ = ["regression_wf"]
# %%

##########################################################################################################################################
# regression wf
##########################################################################################################################################

############################################################################################
# setup regression wf
############################################################################################

# %%
##############################################
# regressor regressor timeshift and beta
##############################################

##--##--##--##--##--##--##--##--##--##--##--##
# align and downsample wf
##--##--##--##--##--##--##--##--##--##--##--##

# align regressor
align_regressor_timeseries = AlignTimeSeries(description="align regressor timeseries")
# downsample
down_sample_ref_timeseries = DownsampleTimeSeries(description="down sample ref timeseries")
down_sample_align_timeseries = DownsampleTimeSeries(description="down sample align timeseries")

align_downsample_wf = ProcessWorkflow(
    (
        # downsample ref timeseries
        (ProcessWorkflow.input.down_sampling_factor, down_sample_ref_timeseries.input.down_sampling_factor),
        (ProcessWorkflow.input.ref_timeseries, down_sample_ref_timeseries.input.timeseries),
        (down_sample_ref_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_ref_timeseries),
        # align align timeseries
        (ProcessWorkflow.input.align_timeseries, align_regressor_timeseries.input.timeseries),
        (ProcessWorkflow.input.timeshift, align_regressor_timeseries.input.timeshift),
        (ProcessWorkflow.input.sample_time, align_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.ref_timeseries.shape[0], align_regressor_timeseries.input.length),
        (ValueNode(True).output.value, align_regressor_timeseries.input.fill_nan),
        # down sample aligned timeseries
        (ProcessWorkflow.input.down_sampling_factor, down_sample_align_timeseries.input.down_sampling_factor),
        (align_regressor_timeseries.output.aligned_timeseries, down_sample_align_timeseries.input.timeseries),
        (down_sample_align_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_aligned_timeseries)
    ),
    description="align and down sample workflow"
)

# %%
# regress regressor and regressor
global_regressor_correlate = Correlate(description="regressor regressor regress")
global_regressor_align_downsample = align_downsample_wf.copy(description = "regressor regressor align downsample")
global_regressor_regression = RegressCVR(description="regressor regressor regression")

global_regressor_regression_wf = ProcessWorkflow(
    (
        # correlate 
        (ProcessWorkflow.input.global_timeseries, global_regressor_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.regressor_timeseries, global_regressor_correlate.input.signal_timeseries_b),
        (ProcessWorkflow.input.sample_time, global_regressor_correlate.input.time_step),
        (ProcessWorkflow.input.correlation_window, global_regressor_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, global_regressor_correlate.input.phat),
        (ProcessWorkflow.input.correlation_peak_threshold, global_regressor_correlate.input.peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, global_regressor_correlate.input.multi_peak_strategy),
        (ProcessWorkflow.input.reference_regressor_timeshift, global_regressor_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_regressor_correlate.input.bipolar),
        (ProcessWorkflow.input.align_regressor_lower_bound, global_regressor_correlate.input.lower_limit),
        (ProcessWorkflow.input.align_regressor_upper_bound, global_regressor_correlate.input.upper_limit),
        (global_regressor_correlate.output.timeshift_maxcorr, ProcessWorkflow.output.global_regressor_timeshift_maxcorr),
        (global_regressor_correlate.output.maxcorr, ProcessWorkflow.output.global_regressor_maxcorr),
        (global_regressor_correlate.output.timeshifts, ProcessWorkflow.output.global_regressor_timeshifts),
        (global_regressor_correlate.output.correlations, ProcessWorkflow.output.global_regressor_correlations),
        # align downsample
        (ProcessWorkflow.input._, global_regressor_align_downsample.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.regressor_timeseries, global_regressor_align_downsample.input.align_timeseries),
        (ProcessWorkflow.input.global_timeseries, global_regressor_align_downsample.input.ref_timeseries),
        (global_regressor_correlate.output.timeshift_maxcorr, global_regressor_align_downsample.input.timeshift),
        (global_regressor_align_downsample.output.down_sampled_ref_timeseries, ProcessWorkflow.output.down_sampled_global_timeseries),
        (global_regressor_align_downsample.output.down_sampled_aligned_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_regressor_timeseries),
        # regress
        (ProcessWorkflow.input.down_sampled_regression_confounds_df, global_regressor_regression.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_regressor_regression.input.confound_regressor_correlation_threshold),
        (global_regressor_align_downsample.output.down_sampled_ref_timeseries, global_regressor_regression.input.dv_ts),
        (global_regressor_align_downsample.output.down_sampled_aligned_timeseries, global_regressor_regression.input.regressor_timeseries),
        # output
        (global_regressor_regression.output.regressor_beta, ProcessWorkflow.output.global_regressor_beta),
        (global_regressor_regression.output.predictions, ProcessWorkflow.output.down_sampled_global_regressor_predictions),
        (global_regressor_regression.output.regressor_p, ProcessWorkflow.output.global_regressor_p_value),
        (global_regressor_regression.output.regressor_se, ProcessWorkflow.output.global_regressor_se),
        (global_regressor_regression.output.regressor_t, ProcessWorkflow.output.global_regressor_t_value),
        (global_regressor_regression.output.dof, ProcessWorkflow.output.global_regressor_dof),
        (global_regressor_regression.output.r_squared, ProcessWorkflow.output.global_regressor_r_squared),
        (global_regressor_regression.output.adjusted_r_squared, ProcessWorkflow.output.global_regressor_adjusted_r_squared),
    ),
    description="global regressor timeshift and beta wf"
)

#%%
############################################################################################
# iterative cvr wf
############################################################################################

# %%
##############################################
# iterate align downsample over depvars timeseries
##############################################
iterate_cvr_find_timeshift = find_timeshift_wf.copy(description="iterate cvr - find timeshift")
##############################################
# iterate align downsample over depvars timeseries
##############################################
iterate_cvr_align_downsample = IteratingNode(align_downsample_wf.copy(), iterating_inputs=("ref_timeseries", "timeshift"), iterating_name="depvars", description="iterate cvr – align and downsample").setDefaultInputs(depvarsIter_nr_parallel_processes = -1)

##############################################
# iterate calculate cvr over depvars timeseries
##############################################
iterate_cvr_regress = IteratingNode(RegressCVR(), iterating_inputs=("dv_ts", "regressor_timeseries"), iterating_name="depvars", exclude_outputs=("design_matrix", "betas"), description="iterate cvr – regress").setDefaultInputs(depvarsIter_nr_parallel_processes = -1)

# %%
iterate_cvr_wf = ProcessWorkflow(
    (
        # find timeshift 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_find_timeshift.input.nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_find_timeshift.input.show_pbar),
        (ProcessWorkflow.input.depvars_timeseries, iterate_cvr_find_timeshift.input.depvars_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_cvr_find_timeshift.input.sample_time),
        (ProcessWorkflow.input.timeseries_masker, iterate_cvr_find_timeshift.input.timeseries_masker),
        (ProcessWorkflow.input.align_regressor_lower_bound, iterate_cvr_find_timeshift.input.align_regressor_lower_bound),
        (ProcessWorkflow.input.align_regressor_upper_bound, iterate_cvr_find_timeshift.input.align_regressor_upper_bound),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_cvr_find_timeshift.input.maxcorr_bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_cvr_find_timeshift.input.correlation_window),
        (ProcessWorkflow.input.correlation_phat, iterate_cvr_find_timeshift.input.correlation_phat),
        (ProcessWorkflow.input.correlation_peak_threshold, iterate_cvr_find_timeshift.input.correlation_peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_cvr_find_timeshift.input.correlation_multi_peak_strategy),
        (ProcessWorkflow.input.filter_timeshifts_size, iterate_cvr_find_timeshift.input.filter_timeshifts_size),
        (ProcessWorkflow.input.filter_timeshifts_smooth_fwhm, iterate_cvr_find_timeshift.input.filter_timeshifts_smooth_fwhm),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, iterate_cvr_find_timeshift.input.filter_timeshifts_filter_type),
        (ProcessWorkflow.input.regressor_timeseries, iterate_cvr_find_timeshift.input.regressor_timeseries),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_cvr_find_timeshift.input.reference_regressor_timeshift),
        (iterate_cvr_find_timeshift.output.all / iterate_cvr_find_timeshift.output.histogram_timeshift_peak, ProcessWorkflow.output._),
        (iterate_cvr_find_timeshift.output.histogram_timeshift_peak, ProcessWorkflow.output.iterate_reference_regressor_timeshift),
        # iterate align downsample 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_align_downsample.input.depvarsIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_align_downsample.input.depvarsIter_show_pbar),
        (ProcessWorkflow.input.depvars_timeseries.T, iterate_cvr_align_downsample.input.depvarsIter_ref_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_cvr_align_downsample.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_cvr_align_downsample.input.down_sampling_factor),
        (ProcessWorkflow.input.regressor_timeseries, iterate_cvr_align_downsample.input.align_timeseries),
        (ValueNode(False).output.value, iterate_cvr_align_downsample.input.depvarsIter_concat_array_outputs),
        (iterate_cvr_find_timeshift.output.depvarsIter_timeshift_maxcorr, iterate_cvr_align_downsample.input.depvarsIter_timeshift),
        (iterate_cvr_align_downsample.output.all / iterate_cvr_align_downsample.output["depvarsIter_down_sampled_ref_timeseries", "depvarsIter_down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (iterate_cvr_align_downsample.output.depvarsIter_down_sampled_ref_timeseries, ProcessWorkflow.output.depvarsIter_down_sampled_depvars_timeseries),
        (iterate_cvr_align_downsample.output.depvarsIter_down_sampled_aligned_timeseries, ProcessWorkflow.output.depvarsIter_down_sampled_aligned_regressor_timeseries),
        # iterate calculate cvr
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_regress.input.depvarsIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_regress.input.depvarsIter_show_pbar),
        (ProcessWorkflow.input.down_sampled_regression_confounds_df, iterate_cvr_regress.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, iterate_cvr_regress.input.confound_regressor_correlation_threshold),
        (ValueNode(False).output.value, iterate_cvr_regress.input.depvarsIter_concat_array_outputs),
        (iterate_cvr_align_downsample.output.depvarsIter_down_sampled_ref_timeseries, iterate_cvr_regress.input.depvarsIter_dv_ts),
        (iterate_cvr_align_downsample.output.depvarsIter_down_sampled_aligned_timeseries, iterate_cvr_regress.input.depvarsIter_regressor_timeseries),
        (iterate_cvr_regress.output.all - iterate_cvr_regress.output.depvarsIter_predictions, ProcessWorkflow.output._),
        (iterate_cvr_regress.output.depvarsIter_predictions, ProcessWorkflow.output.depvarsIter_down_sampled_depvars_predictions)
    ),
    description="iterate cvr wf"
)

# %%

regression_wf = ProcessWorkflow(
    (
        # regression setup
        (ProcessWorkflow.input._, global_regressor_regression_wf.input.all),
        (global_regressor_regression_wf.output.all, ProcessWorkflow.output._),
        # iterative regression
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("sample_time", "timeseries_masker", "down_sampling_factor", "nr_parallel_processes", "show_pbar", "align_regressor_lower_bound", "align_regressor_upper_bound", "maxcorr_bipolar", "correlation_window", "correlation_phat", "correlation_peak_threshold", "correlation_multi_peak_strategy", "filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "confound_regressor_correlation_threshold", "depvars_timeseries")]),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_cvr_wf.input.reference_regressor_timeshift),
        (ProcessWorkflow.input.regressor_timeseries, iterate_cvr_wf.input.regressor_timeseries),
        (ProcessWorkflow.input.down_sampled_regression_confounds_df, iterate_cvr_wf.input.down_sampled_regression_confounds_df),
        (iterate_cvr_wf.output.all / iterate_cvr_wf.output.iterate_reference_regressor_timeshift, ProcessWorkflow.output._),
        (iterate_cvr_wf.output.iterate_reference_regressor_timeshift, ProcessWorkflow.output.reference_regressor_timeshift),
    ),
    description="regression wf"
)
# %%
