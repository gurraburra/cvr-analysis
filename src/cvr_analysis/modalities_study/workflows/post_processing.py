# %%
import numpy as np

# process control
from process_control import *
# custom packages
from cvr_analysis.modalities_study.workflows.utils.load_in_data import LoadBOLDData, LoadBidsImg, CropBOLDImg, VoxelTimeSeriesMasker, RoiTimeSeriesMasker, GetTimeSeriesEvent
from cvr_analysis.modalities_study.workflows.utils.signal_processing import NewSampleTime, ResampleTimeSeries, DetrendTimeSeries, TemporalFilterTimeSeries, TimeLimitTimeSeries
from cvr_analysis.modalities_study.workflows.utils.data_computation import Correlate, AlignTimeSeries

# %%

##########################################################################################################################################
# post-processing wf
##########################################################################################################################################

############################################################################################
# data loader wf
############################################################################################

##############################################
# bold data loader
##############################################
bold_loader = LoadBOLDData(description="load in bold data")

##############################################
# co2 data loader
##############################################
co2_loader = GetTimeSeriesEvent()

##############################################
# voxel mask loader
##############################################
voxel_mask_loader = LoadBidsImg(description="load in data mask")
# crop bold img
crop_bold_img = CropBOLDImg(description="crop bold data img")

##############################################
# timeseries masker
##############################################
# voxel data
voxel_masker = VoxelTimeSeriesMasker(description="create voxel timeseries masker")
# roi data
label_masker_loader = LoadBidsImg(description="load labels img")
roi_masker = RoiTimeSeriesMasker(description="create roi timeseries masker")
load_create_roi_masker = ProcessWorkflow(
    (
        # labels masker loader
        (ProcessWorkflow.input._, label_masker_loader.input.all / label_masker_loader.input.suffix),
        (ValueNode("dseg").output.value, label_masker_loader.input.suffix),
        # create timeseries masker
        (ProcessWorkflow.input._, roi_masker.input[("voxel_mask_img", "spatial_smoothing_fwhm")]),
        (label_masker_loader.output.bids_img, roi_masker.input.labels_img),
        (roi_masker.output.roi_masker, ProcessWorkflow.output._),
    ),
    description="load and create roi timeseries masker"     
)

# conditioanlly select timeseries masker -> None : use voxels as timeseries, other use a dseg labels masker
timeseries_masker = ConditionalNode("roi_masker", default_condition=None, 
                                            condition_node_map={
                                                                    None : voxel_masker,
                                                                    ConditionalNode.no_match_condition : load_create_roi_masker
                                                                },
                                            input_mapping={"roi_masker" : load_create_roi_masker.input.desc},
                                            output_mapping={"timeseries_masker" : (voxel_masker.output.voxel_masker, 
                                                                                    load_create_roi_masker.output.roi_masker)},
                                            description="timeseries masker"
                                        )
# get timeseries
get_bold_ts = CustomNode(lambda bold_img, timeseries_masker : timeseries_masker.fit_transform(bold_img), outputs="bold_timeseries", description="get bold timeseries")
generate_bold_times = CustomNode(lambda tr, nr_measurements : np.arange(nr_measurements) * tr, outputs="bold_times", description="generate bold timeseries times")


# %%
data_loader_wf = ProcessWorkflow(
    (
        # bold_data
        (ProcessWorkflow.input._, bold_loader.input.all / bold_loader.input[("load_confounds", "load_events")]),
        (ProcessWorkflow.input.include_confounds, bold_loader.input.load_confounds),
        (ValueNode(True).output.value, bold_loader.input.load_events),
        (bold_loader.output.confounds_df, ProcessWorkflow.output.confounds_df),
        (bold_loader.output.tr,  ProcessWorkflow.output.tr),
        (bold_loader.output.nr_measurements,  ProcessWorkflow.output.nr_measurements),
        # co2 wf
        (bold_loader.output.events_df, co2_loader.input.events_df),
        (ValueNode(["edited_end_tidal_co2", "end_tidal_co2"]).output.value, co2_loader.input.event_name),
        (co2_loader.output.times, ProcessWorkflow.output.co2_times),
        (co2_loader.output.timeseries, ProcessWorkflow.output.co2_timeseries),
        (co2_loader.output.event_name, ProcessWorkflow.output.co2_event_name),
        # voxel mask loader
        (ProcessWorkflow.input._, voxel_mask_loader.input.all / voxel_mask_loader.input[("desc", "suffix")]),
        (ProcessWorkflow.input.voxel_mask, voxel_mask_loader.input.desc),
        (ValueNode("mask").output.value, voxel_mask_loader.input.suffix),
        (voxel_mask_loader.output.bids_img, ProcessWorkflow.output.voxel_mask_img),
        # crop bold data
        (voxel_mask_loader.output.bids_img, crop_bold_img.input.voxel_mask_img),
        (bold_loader.output.bold_img, crop_bold_img.input.bold_img),
        # timeseries masker
        (ProcessWorkflow.input._, timeseries_masker.input.all / timeseries_masker.input.voxel_mask_img),
        (voxel_mask_loader.output.bids_img, timeseries_masker.input.voxel_mask_img),
        (timeseries_masker.output.timeseries_masker, ProcessWorkflow.output._),
        # get timeseries
        (crop_bold_img.output.cropped_bold_img, get_bold_ts.input.bold_img),
        (timeseries_masker.output.timeseries_masker, get_bold_ts.input.timeseries_masker),
        (get_bold_ts.output.bold_timeseries, ProcessWorkflow.output.bold_timeseries),
        # generate_bold_times
        (bold_loader.output.tr, generate_bold_times.input.tr),
        (bold_loader.output.nr_measurements, generate_bold_times.input.nr_measurements),
        (generate_bold_times.output.bold_times, ProcessWorkflow.output.bold_times)
    ),
    description="data loader workflow"
).setDefaultInputs(include_confounds = True)
# %%

############################################################################################
# signal processing wf
############################################################################################

##############################################
# resample time series
##############################################
# calc up sampling time
new_sampling_time = NewSampleTime(description="calculate new sample time and upsampling factor")
resample_bold_timeseries = ResampleTimeSeries(description="resample bold timeseries")
resample_confounds_df = ResampleTimeSeries(description="resample confounds df")
resample_co2_series = ResampleTimeSeries(description="resample co2 series")

# resample time wf
resample_wf = ProcessWorkflow(
    (
        # new sampling time
        (ProcessWorkflow.input.min_sample_freq, new_sampling_time.input.min_sample_freq),
        (ProcessWorkflow.input.bold_tr, new_sampling_time.input.old_sample_time),
        (new_sampling_time.output.all, ProcessWorkflow.output._),
        # resample bold timeseries
        (ProcessWorkflow.input.bold_times, resample_bold_timeseries.input.times),
        (ProcessWorkflow.input.bold_timeseries, resample_bold_timeseries.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_bold_timeseries.input.sample_time),
        (resample_bold_timeseries.output.resampled_times, ProcessWorkflow.output.resampled_bold_times),
        (resample_bold_timeseries.output.resampled_timeseries, ProcessWorkflow.output.resampled_bold_timeseries),
        # resample confounds
        (ProcessWorkflow.input.bold_times, resample_confounds_df.input.times),
        (ProcessWorkflow.input.confounds_df, resample_confounds_df.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_confounds_df.input.sample_time),
        (resample_confounds_df.output.resampled_times, ProcessWorkflow.output.resampled_confounds_df_times),
        (resample_confounds_df.output.resampled_timeseries, ProcessWorkflow.output.resampled_confounds_df),
        # resample co2 timeseries
        (ProcessWorkflow.input.co2_times, resample_co2_series.input.times),
        (ProcessWorkflow.input.co2_timeseries, resample_co2_series.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_co2_series.input.sample_time),
        (resample_co2_series.output.resampled_times, ProcessWorkflow.output.resampled_co2_times),
        (resample_co2_series.output.resampled_timeseries, ProcessWorkflow.output.resampled_co2_timeseries),
    ),
    description="resampling workflow"
)

# %%
##############################################
# align co2 to global
##############################################
# global co2 correlate
global_co2_correlate = Correlate(description="global co2 correlate")
# global co2 align
global_co2_align = AlignTimeSeries(description="global align co2")
global_co2_align_wf = ProcessWorkflow(
    (
        # global align co2 
        (ProcessWorkflow.input.sample_time, global_co2_correlate.input.time_step),
        (ProcessWorkflow.input.global_align_co2_lower_bound, global_co2_correlate.input.lower_limit),
        (ProcessWorkflow.input.global_align_co2_upper_bound, global_co2_correlate.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, global_co2_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, global_co2_correlate.input.phat),
        (ValueNode(0.0).output.value, global_co2_correlate.input.peak_threshold),
        (ValueNode("max").output.value, global_co2_correlate.input.multi_peak_strategy),
        (ValueNode(None).output.value, global_co2_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_co2_correlate.input.bipolar),
        (ProcessWorkflow.input.bold_timeseries.mean(axis = 1) - ProcessWorkflow.input.bold_timeseries.mean() , global_co2_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.co2_timeseries - ProcessWorkflow.input.co2_timeseries.mean(), global_co2_correlate.input.signal_timeseries_b),
        # align co2 to global
        (ProcessWorkflow.input.sample_time, global_co2_align.input.time_step),
        (ProcessWorkflow.input.bold_timeseries.shape[0], global_co2_align.input.length),
        (ProcessWorkflow.input.co2_timeseries, global_co2_align.input.timeseries),
        (global_co2_correlate.output.timeshift_maxcorr, global_co2_align.input.timeshift),
        (ValueNode(False).output.value, global_co2_align.input.fill_nan),
        (global_co2_align.output.aligned_timeseries, ProcessWorkflow.output.global_aligned_co2_timeseries),
    ), description="global co2 align wf"
)

# %%
##############################################
# time limit timeseries
##############################################
time_limit_bold_timeseries = TimeLimitTimeSeries(description="time_limit bold timeseries")
time_limit_confounds_df = TimeLimitTimeSeries(description="time_limit confounds df")
time_limit_co2_series = TimeLimitTimeSeries(description="time_limit co2 series")

# %%
# time_limit time wf
time_limit_wf = ProcessWorkflow(
    (
        # time_limit bold timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_bold_timeseries.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_bold_timeseries.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_bold_timeseries.input.sample_time),
        (ProcessWorkflow.input.bold_timeseries, time_limit_bold_timeseries.input.timeseries),
        (time_limit_bold_timeseries.output.limited_times, ProcessWorkflow.output.time_limited_bold_times),
        (time_limit_bold_timeseries.output.limited_timeseries, ProcessWorkflow.output.time_limited_bold_timeseries),
        # time_limit confounds
        (ProcessWorkflow.input.analysis_start_time, time_limit_confounds_df.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_confounds_df.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_confounds_df.input.sample_time),
        (ProcessWorkflow.input.confounds_df, time_limit_confounds_df.input.timeseries),
        (time_limit_confounds_df.output.limited_times, ProcessWorkflow.output.time_limited_confounds_df_times),
        (time_limit_confounds_df.output.limited_timeseries, ProcessWorkflow.output.time_limited_confounds_df),
        # time_limit co2 timeseries
        (ProcessWorkflow.input.analysis_start_time, time_limit_co2_series.input.start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_co2_series.input.end_time),
        (ProcessWorkflow.input.sample_time, time_limit_co2_series.input.sample_time),
        (ProcessWorkflow.input.co2_timeseries, time_limit_co2_series.input.timeseries),
        (time_limit_co2_series.output.limited_times, ProcessWorkflow.output.time_limited_co2_times),
        (time_limit_co2_series.output.limited_timeseries, ProcessWorkflow.output.time_limited_co2_timeseries),
    ),
    description="time limit workflow"
)

# %%
##############################################
# detrend timeseries
##############################################
detrend_bold_timeseries = DetrendTimeSeries(description="detrend bold timeseries")
detrend_confounds_df = DetrendTimeSeries(description="detrend confounds df")
detrend_co2_series = DetrendTimeSeries(description="detrend co2 series")

# detrend time wf
detrend_wf = ProcessWorkflow(
    (
        # detrend bold timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_bold_timeseries.input.linear_order),
        (ProcessWorkflow.input.bold_timeseries, detrend_bold_timeseries.input.timeseries),
        (detrend_bold_timeseries.output.detrended_timeseries, ProcessWorkflow.output.detrended_bold_timeseries),
        # detrend confounds
        (ProcessWorkflow.input.detrend_linear_order, detrend_confounds_df.input.linear_order),
        (ProcessWorkflow.input.confounds_df, detrend_confounds_df.input.timeseries),
        (detrend_confounds_df.output.detrended_timeseries, ProcessWorkflow.output.detrended_confounds_df),
        # detrend co2 timeseries
        (ProcessWorkflow.input.detrend_linear_order, detrend_co2_series.input.linear_order),
        (ProcessWorkflow.input.co2_timeseries, detrend_co2_series.input.timeseries),
        (detrend_co2_series.output.detrended_timeseries, ProcessWorkflow.output.detrended_co2_timeseries),
    ),
    description="detrend workflow"
)

# %%
##############################################
# temporal_filter timeseries
##############################################
temporal_filter_bold_timeseries = TemporalFilterTimeSeries(description="temporal filter bold timeseries")
temporal_filter_confounds_df = TemporalFilterTimeSeries(description="temporal filter confounds df")
temporal_filter_co2_series = TemporalFilterTimeSeries(description="temporal filter co2 series")

# filter order
filter_order = ValueNode(6)

# temporal_filter time wf
temporal_filter_wf = ProcessWorkflow(
    (
        # temporal_filter bold timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_bold_timeseries.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_bold_timeseries.input.filter_freq),
        (ProcessWorkflow.input.bold_timeseries, temporal_filter_bold_timeseries.input.timeseries),
        (filter_order.output.value, temporal_filter_bold_timeseries.input.filter_order),
        (temporal_filter_bold_timeseries.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_bold_timeseries),
        # temporal_filter confounds
        (ProcessWorkflow.input.sample_time, temporal_filter_confounds_df.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_confounds_df.input.filter_freq),
        (ProcessWorkflow.input.confounds_df, temporal_filter_confounds_df.input.timeseries),
        (filter_order.output.value, temporal_filter_confounds_df.input.filter_order),
        (temporal_filter_confounds_df.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_confounds_df),
        # temporal_filter co2 timeseries
        (ProcessWorkflow.input.sample_time, temporal_filter_co2_series.input.sample_time),
        (ProcessWorkflow.input.temporal_filter_freq, temporal_filter_co2_series.input.filter_freq),
        (ProcessWorkflow.input.co2_timeseries, temporal_filter_co2_series.input.timeseries),
        (filter_order.output.value, temporal_filter_co2_series.input.filter_order),
        (temporal_filter_co2_series.output.temporal_filtered_timeseries, ProcessWorkflow.output.temporal_filtered_co2_timeseries),
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
        # global align co2
        (ProcessWorkflow.input._, global_co2_align_wf.input[("correlation_phat", "correlation_window", "global_align_co2_lower_bound", "global_align_co2_upper_bound")]),
        (resample_wf.output.resampled_bold_timeseries, global_co2_align_wf.input.bold_timeseries),
        (resample_wf.output.resampled_co2_timeseries, global_co2_align_wf.input.co2_timeseries),
        (resample_wf.output.new_sample_time, global_co2_align_wf.input.sample_time),
        # time limit wf
        (ProcessWorkflow.input.analysis_start_time, time_limit_wf.input.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_wf.input.analysis_end_time),
        (resample_wf.output.new_sample_time, time_limit_wf.input.sample_time),
        (resample_wf.output.resampled_bold_timeseries, time_limit_wf.input.bold_timeseries),
        (resample_wf.output.resampled_confounds_df, time_limit_wf.input.confounds_df),
        (global_co2_align_wf.output.global_aligned_co2_timeseries, time_limit_wf.input.co2_timeseries),
        # detrend wf
        (ProcessWorkflow.input.detrend_linear_order, detrend_wf.input.detrend_linear_order),
        (time_limit_wf.output.time_limited_bold_timeseries, detrend_wf.input.bold_timeseries),
        (time_limit_wf.output.time_limited_confounds_df, detrend_wf.input.confounds_df),
        (time_limit_wf.output.time_limited_co2_timeseries, detrend_wf.input.co2_timeseries),
        # temporal filter
        (ProcessWorkflow.input._, temporal_filter_wf.input.temporal_filter_freq),
        (resample_wf.output.new_sample_time, temporal_filter_wf.input.sample_time),
        (detrend_wf.output.detrended_bold_timeseries, temporal_filter_wf.input.bold_timeseries),
        (detrend_wf.output.detrended_confounds_df, temporal_filter_wf.input.confounds_df),
        (detrend_wf.output.detrended_co2_timeseries, temporal_filter_wf.input.co2_timeseries),
        (temporal_filter_wf.output.temporal_filtered_bold_timeseries, ProcessWorkflow.output.temporal_filtered_detrended_time_limited_up_sampled_bold_timeseries),
        (temporal_filter_wf.output.temporal_filtered_co2_timeseries, ProcessWorkflow.output.temporal_filtered_detrended_time_limited_global_aligned_up_sampled_co2_timeseries),
        (temporal_filter_wf.output.temporal_filtered_confounds_df, ProcessWorkflow.output.temporal_filtered_detrended_time_limited_up_sampled_confounds_df),
    ),
    description="signal processing workflow"
)


# %%
# post processing wf
post_processing_wf = ProcessWorkflow(
    (
        # data loader wf
        (ProcessWorkflow.input._, data_loader_wf.input.all),
        (data_loader_wf.output.co2_event_name, ProcessWorkflow.output.co2_event_name),
        (data_loader_wf.output.voxel_mask_img, ProcessWorkflow.output.voxel_mask_img),
        (data_loader_wf.output.timeseries_masker, ProcessWorkflow.output.timeseries_masker),
        (data_loader_wf.output.tr, ProcessWorkflow.output.bold_tr),
        # signal processing wf
        (ProcessWorkflow.input._, signal_processing_wf.input[("min_sample_freq", "analysis_start_time", "analysis_end_time", "detrend_linear_order", "temporal_filter_freq", "correlation_phat", "correlation_window", "global_align_co2_lower_bound", "global_align_co2_upper_bound")]),
        (data_loader_wf.output.tr, signal_processing_wf.input.bold_tr),
        (data_loader_wf.output.bold_times, signal_processing_wf.input.bold_times),
        (data_loader_wf.output.bold_timeseries, signal_processing_wf.input.bold_timeseries),
        (data_loader_wf.output.confounds_df, signal_processing_wf.input.confounds_df),
        (data_loader_wf.output.co2_times, signal_processing_wf.input.co2_times),
        (data_loader_wf.output.co2_timeseries, signal_processing_wf.input.co2_timeseries),
        (signal_processing_wf.output.all, ProcessWorkflow.output._),
    ),
    description="post-processing wf"
)

# %%