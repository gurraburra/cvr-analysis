# %%
import numpy as np
from nilearn.masking import compute_epi_mask, apply_mask

# process control
from process_control import *
# custom packages
from cvr_analysis.workflows.utils.load_in_data import LoadBOLDData, LoadBidsImg, CropBOLDImg, VoxelTimeSeriesMasker, RoiTimeSeriesMasker, LoadTimeseriesEvent, LoadPhysioData
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
# bold data loader
##############################################
bold_loader = LoadBOLDData(description="load in bold data")

##############################################
# regressor data loader
##############################################
# regressor type
def regressorType(regressor : str):
    if regressor == "global-signal":
        return "global-signal", 
    elif regressor.startswith("physio:"):
        return "physio", 
    else:
        return "event", 
regressor_type = CustomNode(regressorType, outputs=("regressor_type",))
# split physio name
def splitPhysioName(regressor : str):
    # regressor in the form: physio:<recording>:<variable>:<variable>
    physio_parts = regressor.split(":")
    if len(physio_parts) > 2:
        recording = None if physio_parts[1] == "None" else physio_parts[1]
        return recording, physio_parts[2:]
    else:
        raise ValueError(f"incorrect defined physiological recording '{recording}'")
physio_name_split = CustomNode(splitPhysioName, outputs=("recording", "variables"))
# load physio
physio_loader = LoadPhysioData()
# assert single variables
def singlePhysVar(variables):
    if len(variables) > 1:
        raise ValueError(f"multiple variables specified for regressor")
    else:
        return variables[0],
single_phys_var = CustomNode(singlePhysVar, outputs=("variable",))
physio_wf = ProcessWorkflow(
    (
        # split physio name
        (ProcessWorkflow.input._, physio_name_split.input.regressor),
        # single var
        (physio_name_split.output.variables, single_phys_var.input.variables),
        # physio loader
        (ProcessWorkflow.input._, physio_loader.input.all - physio_loader.input[("recording", "variables")]),
        (physio_name_split.output.recording, physio_loader.input.recording),
        (single_phys_var.output.variable, physio_loader.input.variables),
        (physio_loader.output.all, ProcessWorkflow.output._)
    ), description="load physio data"
)
event_loader = LoadTimeseriesEvent()
#%%
# loader
regressor_ts_loader = ConditionalNode("regressor_type", 
                                  {"global-signal" : None, 
                                   "physio" : physio_wf,
                                   "event" : event_loader}, 
                                   input_mapping =
                                   {
                                       "regressor" : event_loader.input.event_name,
                                   },
                                   output_mapping = 
                                   {
                                       "regressor_times" : (event_loader.output.times,physio_wf.output.times),
                                       "regressor_timeseries" : (event_loader.output.timeseries,physio_wf.output.timeseries), 
                                       "regressor_unit" : (event_loader.output.unit,physio_wf.output.units) 
                                   },
                                   description="conditionally load regressor data").setDefaultInputs(regressor = ProcessNode.no_default_input)

# convert time unit
def convFactorUnit(timeseries, unit, use_mmhg = True):
    if timeseries is not None:
        if unit.lower() == 'kpa':
            if use_mmhg:
                return timeseries*7.50061683, "mmHg"
            else:
                return timeseries*1.0, "kPa"
        elif unit.lower() == 'mmhg':
            if not use_mmhg:
                return 0.133322368, "kPa"
            else:
                return timeseries*1.0, "mmHg"
        else:
            return timeseries*1.0, unit
    else:
        return None, None
conv_factor = CustomNode(convFactorUnit, outputs=("conv_timeseries", "conv_unit"))
regressor_wf = ProcessWorkflow(
    (
        # regressor type
        (ProcessWorkflow.input._, regressor_type.input.regressor),
        # regressor ts loader
        (ProcessWorkflow.input._, regressor_ts_loader.input.all / regressor_ts_loader.input[("regressor_type","data_type")]),
        # (ProcessWorkflow.input.regressor, regressor_ts_loader.input.event_name),
        (ValueNode("func").output.value, regressor_ts_loader.input.data_type),
        (regressor_type.output.regressor_type, regressor_ts_loader.input.regressor_type),
        (regressor_ts_loader.output.regressor_times, ProcessWorkflow.output._),
        # convert unit
        (ProcessWorkflow.input._, conv_factor.input.use_mmhg),
        (regressor_ts_loader.output.regressor_timeseries, conv_factor.input.timeseries),
        (regressor_ts_loader.output.regressor_unit, conv_factor.input.unit),
        (conv_factor.output.conv_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (conv_factor.output.conv_unit, ProcessWorkflow.output.regressor_unit),
    ), description="regressor wf"
)
# %%
##############################################
# voxel mask loader
##############################################
load_mask = LoadBidsImg(description="load in data mask")
compute_epi = CustomNode(lambda bold_img : compute_epi_mask(bold_img), outputs="voxel_mask_img", description="nilearn compute EPI")
# either load existing mask or compute new mask
voxel_mask_loader = ConditionalNode("voxel_mask",
                                    condition_node_map={
                                        ConditionalNode.no_match_condition : load_mask,
                                        "nlEPI" : compute_epi,
                                    },
                                    input_mapping={"voxel_mask" : load_mask.input.desc},
                                    output_mapping={"voxel_mask_img" : load_mask.output.bids_img}, 
                                    description="load voxel mask")
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

# compute global signal for alignment
def globalSignal(bold_img, voxel_mask):
    ts = apply_mask(bold_img, voxel_mask)
    return np.mean(ts, axis = 1), 
comp_global_signal = CustomNode(globalSignal, outputs=("global_timeseries", ))

# select regressor
pass_loaded_regressor = ProcessWorkflow(
    (
        (ProcessWorkflow.input.loaded_times, ProcessWorkflow.output.regressor_times),
        (ProcessWorkflow.input.loaded_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (ProcessWorkflow.input.loaded_unit, ProcessWorkflow.output.regressor_unit),
    )
)
pass_global_regressor = ProcessWorkflow(
    (
        (ProcessWorkflow.input.global_times, ProcessWorkflow.output.regressor_times),
        (ProcessWorkflow.input.global_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (ProcessWorkflow.input.global_unit, ProcessWorkflow.output.regressor_unit),
    )
)
select_regressor = ConditionalNode("regressor", {ConditionalNode.no_match_condition : pass_loaded_regressor, "global-signal" : pass_global_regressor}, description="align regressor regresor only if regressor regressor loaded")

# %%
data_loader_wf = ProcessWorkflow(
    (
        # bold_data
        (ProcessWorkflow.input._, bold_loader.input.all / bold_loader.input[("load_confounds",)]),
        (ProcessWorkflow.input.include_motion_confounds, bold_loader.input.load_confounds),
        (bold_loader.output.confounds_df, ProcessWorkflow.output.confounds_df),
        (bold_loader.output.tr,  ProcessWorkflow.output.tr),
        (bold_loader.output.nr_measurements,  ProcessWorkflow.output.nr_measurements),
        # regressor wf
        (ProcessWorkflow.input._, regressor_wf.input.all),
        # voxel mask loader
        (ProcessWorkflow.input._, voxel_mask_loader.input.all / voxel_mask_loader.input[("suffix", "bold_img")]),
        (ValueNode("mask").output.value, voxel_mask_loader.input.suffix),
        (bold_loader.output.bold_img, voxel_mask_loader.input.bold_img),
        (voxel_mask_loader.output.voxel_mask_img, ProcessWorkflow.output.voxel_mask_img),
        # crop bold data
        (voxel_mask_loader.output.voxel_mask_img, crop_bold_img.input.voxel_mask_img),
        (bold_loader.output.bold_img, crop_bold_img.input.bold_img),
        # timeseries masker
        (ProcessWorkflow.input._, timeseries_masker.input.all / timeseries_masker.input.voxel_mask_img),
        (crop_bold_img.output.resampled_voxel_mask_img, timeseries_masker.input.voxel_mask_img),
        (timeseries_masker.output.timeseries_masker, ProcessWorkflow.output._),
        # get timeseries
        (crop_bold_img.output.cropped_bold_img, get_bold_ts.input.bold_img),
        (timeseries_masker.output.timeseries_masker, get_bold_ts.input.timeseries_masker),
        (get_bold_ts.output.bold_timeseries, ProcessWorkflow.output.bold_timeseries),
        # compute global signal
        (crop_bold_img.output.resampled_voxel_mask_img, comp_global_signal.input.voxel_mask),
        (crop_bold_img.output.cropped_bold_img, comp_global_signal.input.bold_img),
        (comp_global_signal.output.global_timeseries, ProcessWorkflow.output._),
        # generate_bold_times
        (bold_loader.output.tr, generate_bold_times.input.tr),
        (bold_loader.output.nr_measurements, generate_bold_times.input.nr_measurements),
        (generate_bold_times.output.bold_times, ProcessWorkflow.output.bold_times),
        # select regressor
        (ProcessWorkflow.input._, select_regressor.input.regressor),
        (regressor_wf.output.regressor_times, select_regressor.input.loaded_times),
        (regressor_wf.output.regressor_timeseries, select_regressor.input.loaded_timeseries),
        (regressor_wf.output.regressor_unit, select_regressor.input.loaded_unit),
        (generate_bold_times.output.bold_times, select_regressor.input.global_times),
        (comp_global_signal.output.global_timeseries, select_regressor.input.global_timeseries),
        (ValueNode("BOLD").output.value, select_regressor.input.global_unit),
        (select_regressor.output.regressor_times, ProcessWorkflow.output.regressor_times),
        (select_regressor.output.regressor_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (select_regressor.output.regressor_unit, ProcessWorkflow.output.regressor_unit),
    ),
    description="data loader workflow"
).setDefaultInputs(include_motion_confounds = True)
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
resample_regressor_series = ResampleTimeSeries(description="resample regressor series")
resample_global_series = ResampleTimeSeries(description="resample global series")

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
        # resample regressor timeseries
        (ProcessWorkflow.input.regressor_times, resample_regressor_series.input.times),
        (ProcessWorkflow.input.regressor_timeseries, resample_regressor_series.input.timeseries),
        (new_sampling_time.output.new_sample_time, resample_regressor_series.input.sample_time),
        (resample_regressor_series.output.resampled_times, ProcessWorkflow.output.resampled_regressor_times),
        (resample_regressor_series.output.resampled_timeseries, ProcessWorkflow.output.resampled_regressor_timeseries),
        # resample global timeseries
        (ProcessWorkflow.input.bold_times, resample_global_series.input.times),
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
        (ValueNode(0.0).output.value, global_regressor_correlate.input.peak_threshold),
        (ValueNode("max").output.value, global_regressor_correlate.input.multi_peak_strategy),
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
time_limit_bold_timeseries = TimeLimitTimeSeries(description="time_limit bold timeseries")
time_limit_confounds_df = TimeLimitTimeSeries(description="time_limit confounds df")
time_limit_regressor_series = TimeLimitTimeSeries(description="time_limit regressor series")
time_limit_global_series = TimeLimitTimeSeries(description="time_limit global series")

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

# %%
##############################################
# detrend timeseries
##############################################
detrend_bold_timeseries = DetrendTimeSeries(description="detrend bold timeseries")
detrend_confounds_df = DetrendTimeSeries(description="detrend confounds df")
detrend_regressor_series = DetrendTimeSeries(description="detrend regressor series")
detrend_global_series = DetrendTimeSeries(description="detrend global series")

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
temporal_filter_bold_timeseries = TemporalFilterTimeSeries(description="temporal filter bold timeseries")
temporal_filter_confounds_df = TemporalFilterTimeSeries(description="temporal filter confounds df")
temporal_filter_regressor_series = TemporalFilterTimeSeries(description="temporal filter regressor series")
temporal_filter_global_series = TemporalFilterTimeSeries(description="temporal filter global series")

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
signal_processing_wf = ProcessWorkflow(
    (  
        # resample wf
        (ProcessWorkflow.input._, resample_wf.input.all),
        (resample_wf.output.up_sampling_factor, ProcessWorkflow.output.up_sampling_factor),
        (resample_wf.output.new_sample_time, ProcessWorkflow.output.up_sampled_sample_time),
        # detrend wf
        (ProcessWorkflow.input.detrend_linear_order, detrend_wf.input.detrend_linear_order),
        (resample_wf.output.resampled_bold_timeseries, detrend_wf.input.bold_timeseries),
        (resample_wf.output.resampled_confounds_df, detrend_wf.input.confounds_df),
        (resample_wf.output.resampled_regressor_timeseries, detrend_wf.input.regressor_timeseries),
        (resample_wf.output.resampled_global_timeseries, detrend_wf.input.global_timeseries),
        # temporal filter
        (ProcessWorkflow.input._, temporal_filter_wf.input.temporal_filter_freq),
        (resample_wf.output.new_sample_time, temporal_filter_wf.input.sample_time),
        (detrend_wf.output.detrended_bold_timeseries, temporal_filter_wf.input.bold_timeseries),
        (detrend_wf.output.detrended_confounds_df, temporal_filter_wf.input.confounds_df),
        (detrend_wf.output.detrended_regressor_timeseries, temporal_filter_wf.input.regressor_timeseries),
        (detrend_wf.output.detrended_global_timeseries, temporal_filter_wf.input.global_timeseries),
        # global align regressor
        (ProcessWorkflow.input._, global_regressor_align_wf.input[("correlation_phat", "correlation_window", "global_align_regressor_lower_bound", "global_align_regressor_upper_bound")]),
        (resample_wf.output.new_sample_time, global_regressor_align_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_global_timeseries, global_regressor_align_wf.input.global_timeseries),
        (temporal_filter_wf.output.temporal_filtered_regressor_timeseries, global_regressor_align_wf.input.regressor_timeseries),
        (global_regressor_align_wf.output.initial_global_regressor_alignment, ProcessWorkflow.output.initial_global_regressor_alignment),
        # time limit wf
        (ProcessWorkflow.input.analysis_start_time, time_limit_wf.input.analysis_start_time),
        (ProcessWorkflow.input.analysis_end_time, time_limit_wf.input.analysis_end_time),
        (resample_wf.output.new_sample_time, time_limit_wf.input.sample_time),
        (temporal_filter_wf.output.temporal_filtered_bold_timeseries, time_limit_wf.input.bold_timeseries),
        (temporal_filter_wf.output.temporal_filtered_confounds_df, time_limit_wf.input.confounds_df),
        (global_regressor_align_wf.output.global_aligned_regressor_timeseries, time_limit_wf.input.regressor_timeseries),
        (temporal_filter_wf.output.temporal_filtered_global_timeseries, time_limit_wf.input.global_timeseries),
        (time_limit_wf.output.time_limited_bold_timeseries, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_bold_timeseries),
        (time_limit_wf.output.time_limited_confounds_df, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_confounds_df),
        (time_limit_wf.output.time_limited_regressor_timeseries, ProcessWorkflow.output.time_limited_global_aligned_temporal_filtered_detrended_up_sampled_regressor_timeseries),
        (time_limit_wf.output.time_limited_global_timeseries, ProcessWorkflow.output.time_limited_temporal_filtered_detrended_up_sampled_global_timeseries),
    ),
    description="signal processing workflow"
)


# %%
# post processing wf
post_processing_wf = ProcessWorkflow(
    (
        # data loader wf
        (ProcessWorkflow.input._, data_loader_wf.input.all),
        (data_loader_wf.output.regressor_unit, ProcessWorkflow.output.regressor_unit),
        (data_loader_wf.output.voxel_mask_img, ProcessWorkflow.output.voxel_mask_img),
        (data_loader_wf.output.timeseries_masker, ProcessWorkflow.output.timeseries_masker),
        (data_loader_wf.output.tr, ProcessWorkflow.output.bold_tr),
        # signal processing wf
        (ProcessWorkflow.input._, signal_processing_wf.input[("min_sample_freq", "analysis_start_time", "analysis_end_time", "detrend_linear_order", "temporal_filter_freq", "correlation_phat", "correlation_window", "global_align_regressor_lower_bound", "global_align_regressor_upper_bound")]),
        (data_loader_wf.output.tr, signal_processing_wf.input.bold_tr),
        (data_loader_wf.output.bold_times, signal_processing_wf.input.bold_times),
        (data_loader_wf.output.bold_timeseries, signal_processing_wf.input.bold_timeseries),
        (data_loader_wf.output.confounds_df, signal_processing_wf.input.confounds_df),
        (data_loader_wf.output.regressor_times, signal_processing_wf.input.regressor_times),
        (data_loader_wf.output.regressor_timeseries, signal_processing_wf.input.regressor_timeseries),
        (data_loader_wf.output.global_timeseries, signal_processing_wf.input.global_timeseries),
        (signal_processing_wf.output.all, ProcessWorkflow.output._),
    ),
    description="post-processing wf"
)

# %%