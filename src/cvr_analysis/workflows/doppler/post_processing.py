# %%
import numpy as np
from nilearn.masking import compute_epi_mask

# process control
from process_control import *
# custom packages
from cvr_analysis.workflows.utils.load_in_data import LoadDopplerData, GetTimeSeriesEvent
from cvr_analysis.workflows.utils.signal_processing import NewSampleTime, ResampleTimeSeries, DetrendTimeSeries, TemporalFilterTimeSeries, TimeLimitTimeSeries
from cvr_analysis.workflows.utils.data_computation import Correlate, AlignTimeSeries

# %%

##########################################################################################################################################
# post-processing wf
##########################################################################################################################################

############################################################################################
# data loader wf
############################################################################################

##############################################
# doppler data loader
##############################################
doppler_loader = LoadDopplerData(description="load in doppler data")

##############################################
# regressor data loader
##############################################
regressor_loader = GetTimeSeriesEvent()

# %%
data_loader_wf = ProcessWorkflow(
    (
        # doppler_data
        (ProcessWorkflow.input._, doppler_loader.input.all),
        (doppler_loader.output.times, ProcessWorkflow.output.doppler_times),
        (doppler_loader.output.time_unit, ProcessWorkflow.output.doppler_time_unit),
        (doppler_loader.output.mean_tr,  ProcessWorkflow.output.doppler_tr),
        (doppler_loader.output.blood_flow_ts, ProcessWorkflow.output.doppler_timeseries),
        (doppler_loader.output.blood_flow_headers, ProcessWorkflow.output.doppler_headers),
        (doppler_loader.output.blood_flow_units, ProcessWorkflow.output.doppler_units),
        # regressor wf
        (ProcessWorkflow.input.regressor_event_name, regressor_loader.input.event_name),
        (doppler_loader.output.events_df, regressor_loader.input.events_df),
        (regressor_loader.output.times, ProcessWorkflow.output.regressor_times),
        (regressor_loader.output.timeseries, ProcessWorkflow.output.regressor_timeseries),
        (regressor_loader.output.event_name, ProcessWorkflow.output.regressor_event_name),
    ),
    description="data loader workflow"
).setDefaultInputs(regressor_event_name = ["edited_end_tidal_co2", "end_tidal_co2"])
# %%

############################################################################################
# signal processing wf
############################################################################################

##############################################
# resample time series
##############################################
# calc up sampling time
new_sampling_time = NewSampleTime(description="calculate new sample time and upsampling factor")
resample_doppler_timeseries = ResampleTimeSeries(description="resample doppler timeseries")
resample_regressor_series = ResampleTimeSeries(description="resample regressor series")

# resample time wf
resample_wf = ProcessWorkflow(
    (
        # new sampling time
        (ProcessWorkflow.input.min_sample_freq, new_sampling_time.input.min_sample_freq),
        (ProcessWorkflow.input.doppler_tr, new_sampling_time.input.old_sample_time),
        (new_sampling_time.output.all, ProcessWorkflow.output._),
        # resample doppler timeseries
        (ProcessWorkflow.input.doppler_times, resample_doppler_timeseries.input.times),
        (ProcessWorkflow.input.doppler_timeseries, resample_doppler_timeseries.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_doppler_timeseries.input.sample_time),
        (resample_doppler_timeseries.output.resampled_times, ProcessWorkflow.output.resampled_doppler_times),
        (resample_doppler_timeseries.output.resampled_timeseries, ProcessWorkflow.output.resampled_doppler_timeseries),
        # resample regressor timeseries
        (ProcessWorkflow.input.regressor_times, resample_regressor_series.input.times),
        (ProcessWorkflow.input.regressor_timeseries, resample_regressor_series.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_regressor_series.input.sample_time),
        (resample_regressor_series.output.resampled_times, ProcessWorkflow.output.resampled_regressor_times),
        (resample_regressor_series.output.resampled_timeseries, ProcessWorkflow.output.resampled_regressor_timeseries),
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
        (ValueNode(0.0).output.value, global_regressor_correlate.input.peak_threshold),
        (ValueNode("max").output.value, global_regressor_correlate.input.multi_peak_strategy),
        (ValueNode(0.0).output.value, global_regressor_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_regressor_correlate.input.bipolar),
        (ProcessWorkflow.input.doppler_timeseries.mean(axis = 1) - ProcessWorkflow.input.doppler_timeseries.mean() , global_regressor_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.regressor_timeseries - ProcessWorkflow.input.regressor_timeseries.mean(), global_regressor_correlate.input.signal_timeseries_b),
        # align regressor to global
        (ProcessWorkflow.input.sample_time, global_regressor_align.input.time_step),
        (ProcessWorkflow.input.doppler_timeseries.shape[0], global_regressor_align.input.length),
        (ProcessWorkflow.input.regressor_timeseries, global_regressor_align.input.timeseries),
        (global_regressor_correlate.output.timeshift_maxcorr, global_regressor_align.input.timeshift),
        (ValueNode(False).output.value, global_regressor_align.input.fill_nan),
        (global_regressor_align.output.aligned_timeseries, ProcessWorkflow.output.global_aligned_regressor_timeseries),
    ), description="global regressor align wf"
)

# %%
##############################################
# time limit timeseries
##############################################
time_limit_doppler_timeseries = TimeLimitTimeSeries(description="time_limit doppler timeseries")
time_limit_regressor_series = TimeLimitTimeSeries(description="time_limit regressor series")

# %%
# time_limit time wf
time_limit_wf = ProcessWorkflow(
    (
        # time_limit doppler timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_doppler_timeseries.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_doppler_timeseries.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_doppler_timeseries.input.sample_time),
        (ProcessWorkflow.input.doppler_timeseries, time_limit_doppler_timeseries.input.timeseries),
        (time_limit_doppler_timeseries.output.limited_times, ProcessWorkflow.output.time_limited_doppler_times),
        (time_limit_doppler_timeseries.output.limited_timeseries, ProcessWorkflow.output.time_limited_doppler_timeseries),
        # time_limit regressor timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_regressor_series.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_regressor_series.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_regressor_series.input.sample_time),
        (ProcessWorkflow.input.regressor_timeseries, time_limit_regressor_series.input.timeseries),
        (time_limit_regressor_series.output.limited_times, ProcessWorkflow.output.time_limited_regressor_times),
        (time_limit_regressor_series.output.limited_timeseries, ProcessWorkflow.output.time_limited_regressor_timeseries),
    ),
    description="time limit workflow"
)

# %%
##############################################
# detrend timeseries
##############################################
detrend_doppler_timeseries = DetrendTimeSeries(description="detrend doppler timeseries")
detrend_regressor_series = DetrendTimeSeries(description="detrend regressor series")

# detrend time wf
detrend_wf = ProcessWorkflow(
    (
        # detrend doppler timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_doppler_timeseries.input.linear_order),
        (ProcessWorkflow.input.doppler_timeseries, detrend_doppler_timeseries.input.timeseries),
        (detrend_doppler_timeseries.output.detrended_timeseries, ProcessWorkflow.output.detrended_doppler_timeseries),
        # detrend regressor timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_regressor_series.input.linear_order),
        (ProcessWorkflow.input.regressor_timeseries, detrend_regressor_series.input.timeseries),
        (detrend_regressor_series.output.detrended_timeseries, ProcessWorkflow.output.detrended_regressor_timeseries),
    ),
    description="detrend workflow"
)

# %%
##############################################
# temporal_filter timeseries
##############################################
temporal_filter_doppler_timeseries = TemporalFilterTimeSeries(description="temporal filter doppler timeseries")
temporal_filter_regressor_series = TemporalFilterTimeSeries(description="temporal filter regressor series")

# filter order
filter_order = ValueNode(6)

# temporal_filter time wf
temporal_filter_wf = ProcessWorkflow(
    (
        # temporal_filter doppler timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_doppler_timeseries.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_doppler_timeseries.input.filter_freq),
        (ProcessWorkflow.input.doppler_timeseries, temporal_filter_doppler_timeseries.input.timeseries),
        (filter_order.output.value, temporal_filter_doppler_timeseries.input.filter_order),
        (temporal_filter_doppler_timeseries.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_doppler_timeseries),
        # temporal_filter regressor timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_regressor_series.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_regressor_series.input.filter_freq),
        (ProcessWorkflow.input.regressor_timeseries, temporal_filter_regressor_series.input.timeseries),
        (filter_order.output.value, temporal_filter_regressor_series.input.filter_order),
        (temporal_filter_regressor_series.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_regressor_timeseries),
    ),
    description="temporal filter workflow"
)

# %%

# signal processing wf
signal_processing_wf = ProcessWorkflow(
    (  
        # resample wf
        (ProcessWorkflow.input._, resample_wf.input.all),
        (resample_wf.output.up_sampling_factor, ProcessWorkflow.output.up_sampling_factor),
        (resample_wf.output.new_sample_time, ProcessWorkflow.output.up_sampled_sample_time),
        # detrend wf
        (ProcessWorkflow.input.detrend_linear_order, detrend_wf.input.detrend_linear_order),
        (resample_wf.output.resampled_doppler_timeseries, detrend_wf.input.doppler_timeseries),
        (resample_wf.output.resampled_regressor_timeseries, detrend_wf.input.regressor_timeseries),
        # temporal filter
        (ProcessWorkflow.input._, temporal_filter_wf.input.temporal_filter_freq),
        (resample_wf.output.new_sample_time, temporal_filter_wf.input.sample_time),
        (detrend_wf.output.detrended_doppler_timeseries, temporal_filter_wf.input.doppler_timeseries),
        (detrend_wf.output.detrended_regressor_timeseries, temporal_filter_wf.input.regressor_timeseries),
        # global align regressor
        (ProcessWorkflow.input._, global_regressor_align_wf.input[("correlation_phat", "correlation_window", "global_align_regressor_lower_bound", "global_align_regressor_upper_bound")]),
        (resample_wf.output.new_sample_time, global_regressor_align_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_doppler_timeseries, global_regressor_align_wf.input.doppler_timeseries),
        (temporal_filter_wf.output.temporal_filtered_regressor_timeseries, global_regressor_align_wf.input.regressor_timeseries),
        # time limit wf
        (ProcessWorkflow.input.analysis_start_time, time_limit_wf.input.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_wf.input.analysis_end_time),
        (resample_wf.output.new_sample_time, time_limit_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_doppler_timeseries, time_limit_wf.input.doppler_timeseries),
        (global_regressor_align_wf.output.global_aligned_regressor_timeseries, time_limit_wf.input.regressor_timeseries),
        (time_limit_wf.output.time_limited_doppler_timeseries, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_doppler_timeseries),
        (time_limit_wf.output.time_limited_regressor_timeseries, ProcessWorkflow.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries),
    ),
    description="signal processing workflow"
)


# %%
# post processing wf
post_processing_wf = ProcessWorkflow(
    (
        # data loader wf
        (ProcessWorkflow.input._, data_loader_wf.input.all),
        (data_loader_wf.output.regressor_event_name, ProcessWorkflow.output.regressor_event_name),
        (data_loader_wf.output.doppler_tr, ProcessWorkflow.output.doppler_tr),
        (data_loader_wf.output.doppler_headers, ProcessWorkflow.output.doppler_headers),
        (data_loader_wf.output.doppler_units, ProcessWorkflow.output.doppler_units),
        # signal processing wf
        (ProcessWorkflow.input._, signal_processing_wf.input[("min_sample_freq", "analysis_start_time", "analysis_end_time", "detrend_linear_order", "temporal_filter_freq", "correlation_phat", "correlation_window", "global_align_regressor_lower_bound", "global_align_regressor_upper_bound")]),
        (data_loader_wf.output.doppler_tr, signal_processing_wf.input.doppler_tr),
        (data_loader_wf.output.doppler_times, signal_processing_wf.input.doppler_times),
        (data_loader_wf.output.doppler_timeseries, signal_processing_wf.input.doppler_timeseries),
        (data_loader_wf.output.regressor_times, signal_processing_wf.input.regressor_times),
        (data_loader_wf.output.regressor_timeseries, signal_processing_wf.input.regressor_timeseries),
        (signal_processing_wf.output.all, ProcessWorkflow.output._),
    ),
    description="post-processing wf"
).setDefaultInputs(detrend_linear_order = None)

# %%