# %%
# process control
from process_control import *
# custom packages
from cvr_analysis.default.helpers.classes.signal_processing import NewSampleTime, ResampleTimeSeries, DetrendTimeSeries, TemporalFilterTimeSeries, TimeLimitTimeSeries
from cvr_analysis.default.helpers.classes.data_computation import Correlate, AlignTimeSeries

__all__ = ["post_processing_wf"]
# %%

##########################################################################################################################################
# post-processing wf
##########################################################################################################################################

##############################################
# resample time series
##############################################
# calc up sampling time
new_sampling_time = NewSampleTime(description="calculate new sample time and upsampling factor")
resample_depvars_timeseries = ResampleTimeSeries(description="resample depvars timeseries")
resample_confounds_df = ResampleTimeSeries(description="resample confounds df")
resample_regressor_series = ResampleTimeSeries(description="resample regressor series")
resample_global_series = ResampleTimeSeries(description="resample global series")

# resample time wf
resample_wf = ProcessWorkflow(
    (
        # new sampling time
        (ProcessWorkflow.input.min_sample_freq, new_sampling_time.input.min_sample_freq),
        (ProcessWorkflow.input.depvars_tr, new_sampling_time.input.old_sample_time),
        (new_sampling_time.output.all, ProcessWorkflow.output._),
        # resample depvars timeseries
        (ProcessWorkflow.input.depvars_times, resample_depvars_timeseries.input.times),
        (ProcessWorkflow.input.depvars_timeseries, resample_depvars_timeseries.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_depvars_timeseries.input.sample_time),
        (resample_depvars_timeseries.output.resampled_times, ProcessWorkflow.output.resampled_depvars_times),
        (resample_depvars_timeseries.output.resampled_timeseries, ProcessWorkflow.output.resampled_depvars_timeseries),
        # resample confounds
        (ProcessWorkflow.input.depvars_times, resample_confounds_df.input.times),
        (ProcessWorkflow.input.confounds_df, resample_confounds_df.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_confounds_df.input.sample_time),
        (resample_confounds_df.output.resampled_times, ProcessWorkflow.output.resampled_confounds_df_times),
        (resample_confounds_df.output.resampled_timeseries, ProcessWorkflow.output.resampled_confounds_df),
        # resample regressor timeseries
        (ProcessWorkflow.input.regressor_times, resample_regressor_series.input.times),
        (ProcessWorkflow.input.regressor_timeseries, resample_regressor_series.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_regressor_series.input.sample_time),
        (resample_regressor_series.output.resampled_times, ProcessWorkflow.output.resampled_regressor_times),
        (resample_regressor_series.output.resampled_timeseries, ProcessWorkflow.output.resampled_regressor_timeseries),
        # resample global timeseries
        (ProcessWorkflow.input.depvars_times, resample_global_series.input.times),
        (ProcessWorkflow.input.global_timeseries, resample_global_series.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_global_series.input.sample_time),
        (resample_global_series.output.resampled_times, ProcessWorkflow.output.resampled_global_times),
        (resample_global_series.output.resampled_timeseries, ProcessWorkflow.output.resampled_global_timeseries),
    ),
    description="resampling workflow"
)

# %%
##############################################
# align regressor to global
##############################################
# global regressor correlate
global_regressor_correlate = Correlate(description="global regressor correlate")
# global regressor align
global_regressor_align = AlignTimeSeries(description="global align regressor")
global_regressor_align_wf = ProcessWorkflow(
    (
        # global align regressor 
        (ProcessWorkflow.input.sample_time, global_regressor_correlate.input.time_step),
        (ProcessWorkflow.input.global_align_regressor_lower_bound, global_regressor_correlate.input.lower_limit),
        (ProcessWorkflow.input.global_align_regressor_upper_bound, global_regressor_correlate.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, global_regressor_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, global_regressor_correlate.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, global_regressor_correlate.input.multi_peak_strategy),
        (ProcessWorkflow.input.correlation_peak_threshold, global_regressor_correlate.input.peak_threshold),
        (ValueNode(0.0).output.value, global_regressor_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_regressor_correlate.input.bipolar),
        (ProcessWorkflow.input.global_timeseries - ProcessWorkflow.input.global_timeseries.mean() , global_regressor_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.regressor_timeseries - ProcessWorkflow.input.regressor_timeseries.mean(), global_regressor_correlate.input.signal_timeseries_b),
        (global_regressor_correlate.output.timeshift_maxcorr, ProcessWorkflow.output.initial_global_regressor_alignment),
        # align regressor to global
        (ProcessWorkflow.input.sample_time, global_regressor_align.input.time_step),
        (ProcessWorkflow.input.global_timeseries.shape[0], global_regressor_align.input.length),
        (ProcessWorkflow.input.regressor_timeseries, global_regressor_align.input.timeseries),
        (global_regressor_correlate.output.timeshift_maxcorr, global_regressor_align.input.timeshift),
        (ValueNode(False).output.value, global_regressor_align.input.fill_nan),
        (global_regressor_align.output.aligned_timeseries, ProcessWorkflow.output.global_aligned_regressor_timeseries),
    ), description="global regressor align wf"
)

# cond_global_regressor_align = ConditionalNode("regressor", {ConditionalNode.no_match_condition : global_regressor_align_wf, "global-signal" : None}, description="align regressor regresor only if regressor regressor loaded")
# %%
##############################################
# time limit timeseries
##############################################
time_limit_depvars_timeseries = TimeLimitTimeSeries(description="time_limit depvars timeseries")
time_limit_confounds_df = TimeLimitTimeSeries(description="time_limit confounds df")
time_limit_regressor_series = TimeLimitTimeSeries(description="time_limit regressor series")
time_limit_global_series = TimeLimitTimeSeries(description="time_limit global series")

# %%
# timelimit can be first step after resampling or last step in postproccesing
# initial time_limit time wf
initial_time_limit_wf = ProcessWorkflow(
    (
        # time_limit depvars timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_depvars_timeseries.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_depvars_timeseries.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_depvars_timeseries.input.sample_time),
        (ProcessWorkflow.input.depvars_timeseries, time_limit_depvars_timeseries.input.timeseries),
        (time_limit_depvars_timeseries.output.limited_times, ProcessWorkflow.output.time_limited_depvars_times),
        (time_limit_depvars_timeseries.output.limited_timeseries, ProcessWorkflow.output.time_limited_depvars_timeseries),
        # time_limit confounds
        (ProcessWorkflow.input.analysis_start_time, time_limit_confounds_df.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_confounds_df.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_confounds_df.input.sample_time),
        (ProcessWorkflow.input.confounds_df, time_limit_confounds_df.input.timeseries),
        (time_limit_confounds_df.output.limited_times, ProcessWorkflow.output.time_limited_confounds_df_times),
        (time_limit_confounds_df.output.limited_timeseries, ProcessWorkflow.output.time_limited_confounds_df),
        # time_limit regressor timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_regressor_series.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_regressor_series.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_regressor_series.input.sample_time),
        (ProcessWorkflow.input.regressor_timeseries, time_limit_regressor_series.input.timeseries),
        (time_limit_regressor_series.output.limited_times, ProcessWorkflow.output.time_limited_regressor_times),
        (time_limit_regressor_series.output.limited_timeseries, ProcessWorkflow.output.time_limited_regressor_timeseries),
        # time_limit global timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_global_series.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_global_series.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_global_series.input.sample_time),
        (ProcessWorkflow.input.global_timeseries, time_limit_global_series.input.timeseries),
        (time_limit_global_series.output.limited_times, ProcessWorkflow.output.time_limited_global_times),
        (time_limit_global_series.output.limited_timeseries, ProcessWorkflow.output.time_limited_global_timeseries),
    ),
    description="time limit workflow"
)
# after
late_time_limit_wf = initial_time_limit_wf.copy()
# no timelimit
time_limit_times = ProcessWorkflow(
    (
        (ProcessWorkflow.input.analysis_start_time, ProcessWorkflow.output.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, ProcessWorkflow.output.analysis_end_time),
    ), description="time_limits_times"
)
# pass None for no timelimte
cond_init_time_limit_times = ConditionalNode("initial_time_limit",{True : time_limit_times, False : None})
cond_late_time_limit_times = ConditionalNode("late_time_limit",{True : time_limit_times, False : None})
# %%
##############################################
# detrend timeseries
##############################################
detrend_depvars_timeseries = DetrendTimeSeries(description="detrend depvars timeseries")
detrend_confounds_df = DetrendTimeSeries(description="detrend confounds df")
detrend_regressor_series = DetrendTimeSeries(description="detrend regressor series")
detrend_global_series = DetrendTimeSeries(description="detrend global series")

# detrend time wf
detrend_wf = ProcessWorkflow(
    (
        # detrend depvars timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_depvars_timeseries.input.linear_order),
        (ProcessWorkflow.input.depvars_timeseries, detrend_depvars_timeseries.input.timeseries),
        (detrend_depvars_timeseries.output.detrended_timeseries, ProcessWorkflow.output.detrended_depvars_timeseries),
        # detrend confounds
        (ProcessWorkflow.input.detrend_linear_order, detrend_confounds_df.input.linear_order),
        (ProcessWorkflow.input.confounds_df, detrend_confounds_df.input.timeseries),
        (detrend_confounds_df.output.detrended_timeseries, ProcessWorkflow.output.detrended_confounds_df),
        # detrend regressor timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_regressor_series.input.linear_order),
        (ProcessWorkflow.input.regressor_timeseries, detrend_regressor_series.input.timeseries),
        (detrend_regressor_series.output.detrended_timeseries, ProcessWorkflow.output.detrended_regressor_timeseries),
        # detrend global timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_global_series.input.linear_order),
        (ProcessWorkflow.input.global_timeseries, detrend_global_series.input.timeseries),
        (detrend_global_series.output.detrended_timeseries, ProcessWorkflow.output.detrended_global_timeseries),
    ),
    description="detrend workflow"
)

# %%
##############################################
# temporal_filter timeseries
##############################################
temporal_filter_depvars_timeseries = TemporalFilterTimeSeries(description="temporal filter depvars timeseries")
temporal_filter_confounds_df = TemporalFilterTimeSeries(description="temporal filter confounds df")
temporal_filter_regressor_series = TemporalFilterTimeSeries(description="temporal filter regressor series")
temporal_filter_global_series = TemporalFilterTimeSeries(description="temporal filter global series")

# filter order
filter_order = ValueNode(6)

# temporal_filter time wf
temporal_filter_wf = ProcessWorkflow(
    (
        # temporal_filter depvars timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_depvars_timeseries.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_depvars_timeseries.input.filter_freq),
        (ProcessWorkflow.input.depvars_timeseries, temporal_filter_depvars_timeseries.input.timeseries),
        (filter_order.output.value, temporal_filter_depvars_timeseries.input.filter_order),
        (temporal_filter_depvars_timeseries.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_depvars_timeseries),
        # temporal_filter confounds
        (ProcessWorkflow.input.sample_time, temporal_filter_confounds_df.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_confounds_df.input.filter_freq),
        (ProcessWorkflow.input.confounds_df, temporal_filter_confounds_df.input.timeseries),
        (filter_order.output.value, temporal_filter_confounds_df.input.filter_order),
        (temporal_filter_confounds_df.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_confounds_df),
        # temporal_filter regressor timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_regressor_series.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_regressor_series.input.filter_freq),
        (ProcessWorkflow.input.regressor_timeseries, temporal_filter_regressor_series.input.timeseries),
        (filter_order.output.value, temporal_filter_regressor_series.input.filter_order),
        (temporal_filter_regressor_series.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_regressor_timeseries),
        # temporal_filter global timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_global_series.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_global_series.input.filter_freq),
        (ProcessWorkflow.input.global_timeseries, temporal_filter_global_series.input.timeseries),
        (filter_order.output.value, temporal_filter_global_series.input.filter_order),
        (temporal_filter_global_series.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_global_timeseries),
    ),
    description="temporal filter workflow"
)

# %%

# signal processing wf
post_processing_wf = ProcessWorkflow(
    (  
        # resample wf
        (ProcessWorkflow.input._, resample_wf.input.all),
        (resample_wf.output.up_sampling_factor, ProcessWorkflow.output.up_sampling_factor),
        (resample_wf.output.new_sample_time, ProcessWorkflow.output.up_sampled_sample_time),
        # initial timelimits
        (ProcessWorkflow.input.initial_time_limit, cond_init_time_limit_times.input.initial_time_limit),
        (ProcessWorkflow.input.analysis_start_time, cond_init_time_limit_times.input.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, cond_init_time_limit_times.input.analysis_end_time),
        # initial time limit wf
        (cond_init_time_limit_times.output.analysis_start_time, initial_time_limit_wf.input.analysis_start_time),
        (cond_init_time_limit_times.output.analysis_end_time, initial_time_limit_wf.input.analysis_end_time),
        (resample_wf.output.new_sample_time, initial_time_limit_wf.input.sample_time),
        (resample_wf.output.resampled_depvars_timeseries, initial_time_limit_wf.input.depvars_timeseries),
        (resample_wf.output.resampled_confounds_df, initial_time_limit_wf.input.confounds_df),
        (resample_wf.output.resampled_regressor_timeseries, initial_time_limit_wf.input.regressor_timeseries),
        (resample_wf.output.resampled_global_timeseries, initial_time_limit_wf.input.global_timeseries),
        # detrend wf
        (ProcessWorkflow.input.detrend_linear_order, detrend_wf.input.detrend_linear_order),
        (initial_time_limit_wf.output.time_limited_depvars_timeseries, detrend_wf.input.depvars_timeseries),
        (initial_time_limit_wf.output.time_limited_confounds_df, detrend_wf.input.confounds_df),
        (initial_time_limit_wf.output.time_limited_regressor_timeseries, detrend_wf.input.regressor_timeseries),
        (initial_time_limit_wf.output.time_limited_global_timeseries, detrend_wf.input.global_timeseries),
        # temporal filter
        (ProcessWorkflow.input._, temporal_filter_wf.input.temporal_filter_freq),
        (resample_wf.output.new_sample_time, temporal_filter_wf.input.sample_time),
        (detrend_wf.output.detrended_depvars_timeseries, temporal_filter_wf.input.depvars_timeseries),
        (detrend_wf.output.detrended_confounds_df, temporal_filter_wf.input.confounds_df),
        (detrend_wf.output.detrended_regressor_timeseries, temporal_filter_wf.input.regressor_timeseries),
        (detrend_wf.output.detrended_global_timeseries, temporal_filter_wf.input.global_timeseries),
        # global align regressor
        (ProcessWorkflow.input._, global_regressor_align_wf.input[("correlation_phat", "correlation_window", "correlation_peak_threshold", "correlation_multi_peak_strategy", "global_align_regressor_lower_bound", "global_align_regressor_upper_bound")]),
        (resample_wf.output.new_sample_time, global_regressor_align_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_global_timeseries, global_regressor_align_wf.input.global_timeseries),
        (temporal_filter_wf.output.temporal_filtered_regressor_timeseries, global_regressor_align_wf.input.regressor_timeseries),
        (global_regressor_align_wf.output.initial_global_regressor_alignment, ProcessWorkflow.output.initial_global_regressor_alignment),
        # late timelimits
        (not_* ProcessWorkflow.input.initial_time_limit, cond_late_time_limit_times.input.late_time_limit),
        (ProcessWorkflow.input.analysis_start_time, cond_late_time_limit_times.input.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, cond_late_time_limit_times.input.analysis_end_time),
        # late time limit wf
        (cond_late_time_limit_times.output.analysis_start_time, late_time_limit_wf.input.analysis_start_time),
        (cond_late_time_limit_times.output.analysis_end_time, late_time_limit_wf.input.analysis_end_time),
        (resample_wf.output.new_sample_time, late_time_limit_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_depvars_timeseries, late_time_limit_wf.input.depvars_timeseries),
        (temporal_filter_wf.output.temporal_filtered_confounds_df, late_time_limit_wf.input.confounds_df),
        (global_regressor_align_wf.output.global_aligned_regressor_timeseries, late_time_limit_wf.input.regressor_timeseries),
        (temporal_filter_wf.output.temporal_filtered_global_timeseries, late_time_limit_wf.input.global_timeseries),
        (late_time_limit_wf.output.time_limited_depvars_timeseries, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_depvars_timeseries),
        (late_time_limit_wf.output.time_limited_confounds_df, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_confounds_df),
        (late_time_limit_wf.output.time_limited_regressor_timeseries, ProcessWorkflow.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries),
        (late_time_limit_wf.output.time_limited_global_timeseries, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_global_timeseries),
    ),
    description="post-processing wf"
).setDefaultInputs(initial_time_limit = False)

# %%