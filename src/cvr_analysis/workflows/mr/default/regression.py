# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.workflows.utils.signal_processing import DownsampleTimeSeries, MaskTimeSeries, DTW
from cvr_analysis.workflows.utils.data_computation import Correlate, AlignTimeSeries, RegressCVR, PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries, RMSTimeSeries, FilterTimeshifts, PCAReducedTimeSeries, HistPeak
from cvr_analysis.workflows.utils.dev.confounds import MotionConfounds, SpikeConfounds, DriftConfounds

# %%

##########################################################################################################################################
# regression wf
##########################################################################################################################################

############################################################################################
# setup regression wf
############################################################################################

##############################################
# convert timeseries it to signals
##############################################

# get baseline
# get timeseries signal values
bold_baseline = BaselineTimeSeries(description="baseline bold timeseries")
global_baseline = BaselineTimeSeries(description="baseline global timeseries")
# get timeseries signal values
bold_percentage = PercentageChangeTimeSeries(description="percentage bold timeseries")
global_percentage = PercentageChangeTimeSeries(description="percentage global timeseries")
confounds_std = StandardizeTimeSeries(description="standardize confounds")

# regressor
regressor_baseline = BaselineTimeSeries(description="baseline regressor timeseries")
demean_signal_wf = ProcessWorkflow(
    (
        (ProcessWorkflow.input.regressor_timeseries, regressor_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, regressor_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, regressor_baseline.input.time_step),
        (ProcessWorkflow.input.regressor_unit, ProcessWorkflow.output.regressor_signal_unit),
        (ProcessWorkflow.input.regressor_timeseries - regressor_baseline.output.baseline, ProcessWorkflow.output.regressor_signal_timeseries),
    ),
    description="create regressor signal"
)
regressor_psc = PercentageChangeTimeSeries(description="percentage regressor")
psc_signal_wf = ProcessWorkflow(
    (
        # baseline
        (ProcessWorkflow.input.regressor_timeseries, regressor_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, regressor_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, regressor_baseline.input.time_step),
        # percentage
        (ProcessWorkflow.input.regressor_timeseries, regressor_psc.input.timeseries),
        (regressor_baseline.output.baseline, regressor_psc.input.baseline),
        (regressor_psc.output.percentage_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        # unit
        (ValueNode("%").output.value + ProcessWorkflow.input.regressor_unit, ProcessWorkflow.output.regressor_signal_unit),
    ),
    description="create regressor signal"
)


cond_regressor_signal = ConditionalNode("regressor", 
                                        {ConditionalNode.no_match_condition : demean_signal_wf, "global-signal" : psc_signal_wf}, 
                                        description="conditionally compute regressor signal")


# %%
signal_timeseries_wf = ProcessWorkflow(
    (
        # bold baseline
        (ProcessWorkflow.input.bold_timeseries, bold_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, bold_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, bold_baseline.input.time_step),
        # bold percentage
        (ProcessWorkflow.input.bold_timeseries, bold_percentage.input.timeseries),
        (bold_baseline.output.baseline, bold_percentage.input.baseline),
        (bold_percentage.output.percentage_timeseries, ProcessWorkflow.output.bold_signal_timeseries),
        # global baseline
        (ProcessWorkflow.input.global_timeseries, global_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, global_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, global_baseline.input.time_step),
        # global percentage
        (ProcessWorkflow.input.global_timeseries, global_percentage.input.timeseries),
        (global_baseline.output.baseline, global_percentage.input.baseline),
        (global_percentage.output.percentage_timeseries, ProcessWorkflow.output.global_signal_timeseries),
        # regressor signal
        (ProcessWorkflow.input.regressor, cond_regressor_signal.input.regressor),
        (ProcessWorkflow.input.regressor_timeseries, cond_regressor_signal.input.regressor_timeseries),
        (ProcessWorkflow.input.regressor_unit, cond_regressor_signal.input.regressor_unit),
        (ProcessWorkflow.input.baseline_strategy, cond_regressor_signal.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, cond_regressor_signal.input.sample_time),
        (cond_regressor_signal.output.regressor_signal_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        (cond_regressor_signal.output.regressor_signal_unit, ProcessWorkflow.output.regressor_signal_unit),
        # confounds
        (ProcessWorkflow.input.confounds_df, confounds_std.input.timeseries),
        (confounds_std.output.standardized_timeseries, ProcessWorkflow.output.confounds_signal_df),
    ),
    description="convert timeseries into signals"
)

# %%
##############################################
# signal characteristics
##############################################
# power
regressor_rms = RMSTimeSeries(description="regressor rms")
global_rms = RMSTimeSeries(description="global rms")
# cond_regressor_rms = ConditionalNode("regressor", {True : regressor_rms, False : None}, description="conditionally calculate regressor rms")

# autocorrelation
regressor_autocorrelation = Correlate(description="regressor timeseries autocorrelation")
global_autocorrelation = Correlate(description="global timeseries autocorrelation")
# cond_regressor_autocorrelation = ConditionalNode("regressor", {True : regressor_autocorrelation, False : None}, description="conditionally calculate regressor autocorrelations")

# %%
signal_characterics_wf = ProcessWorkflow(
        (
            # regressor signal rms
            (ProcessWorkflow.input.regressor_signal_timeseries, regressor_rms.input.timeseries),
            (regressor_rms.output.rms, ProcessWorkflow.output.regressor_signal_rms),
            # regressor autocorrelation
            (ProcessWorkflow.input.correlation_window, regressor_autocorrelation.input.window),
            (ProcessWorkflow.input.correlation_phat, regressor_autocorrelation.input.phat),
            (ProcessWorkflow.input.sample_time, regressor_autocorrelation.input.time_step),
            (ProcessWorkflow.input.regressor_signal_timeseries, regressor_autocorrelation.input[("signal_timeseries_a", "signal_timeseries_b")]),
            (ValueNode(None).output.value, regressor_autocorrelation.input.peak_threshold),
            (ValueNode(None).output.value, regressor_autocorrelation.input.multi_peak_strategy),
            (ValueNode(0).output.value, regressor_autocorrelation.input.ref_timeshift),
            (ValueNode(None).output.value, regressor_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, regressor_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, regressor_autocorrelation.input.bipolar),
            (regressor_autocorrelation.output.timeshifts, ProcessWorkflow.output.regressor_signal_autocorrelation_timeshifts),
            (regressor_autocorrelation.output.correlations, ProcessWorkflow.output.regressor_signal_autocorrelation_correlations),
            # regressor signal rms
            (ProcessWorkflow.input.global_signal_timeseries, global_rms.input.timeseries),
            (global_rms.output.rms, ProcessWorkflow.output.global_signal_rms),
            # regressor autocorrelation
            (ProcessWorkflow.input.correlation_window, global_autocorrelation.input.window),
            (ProcessWorkflow.input.correlation_phat, global_autocorrelation.input.phat),
            (ProcessWorkflow.input.sample_time, global_autocorrelation.input.time_step),
            (ProcessWorkflow.input.global_signal_timeseries, global_autocorrelation.input[("signal_timeseries_a", "signal_timeseries_b")]),
            (ValueNode(None).output.value, global_autocorrelation.input.peak_threshold),
            (ValueNode(None).output.value, global_autocorrelation.input.multi_peak_strategy),
            (ValueNode(0).output.value, global_autocorrelation.input.ref_timeshift),
            (ValueNode(None).output.value, global_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, global_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, global_autocorrelation.input.bipolar),
            (global_autocorrelation.output.timeshifts, ProcessWorkflow.output.global_signal_autocorrelation_timeshifts),
            (global_autocorrelation.output.correlations, ProcessWorkflow.output.global_signal_autocorrelation_correlations),
        ),
        description="signal characteristics"
)
        
# %%
##############################################
# refine regressor
##############################################
##--##--##--##--##--##--##--##--##--##--##--##
# find timeshift
##--##--##--##--##--##--##--##--##--##--##--##
# iterate correlate over bold timeseries
iterate_correlate_wf = IteratingNode(Correlate(), iterating_inputs="signal_timeseries_a", iterating_name="bold", description="iterate correlate bold timeseries").setDefaultInputs(boldIter_nr_parallel_processes = -1)
# filter timeshifts
filter_timeshifts = FilterTimeshifts(description="filter timeshifts")
# histogram peak
hist_peak = HistPeak(description="histogram peak")
# correlation bounds
# add if not None
add_none_lower = CustomNode(lambda x = None, y = None : x + y if x is not None and y is not None else None, description="add none")
add_none_upper = add_none_lower.copy()
# %%
# find timeshift wf
find_timeshift_wf = ProcessWorkflow(
    (
        # correlation bounds
        (ProcessWorkflow.input.align_regressor_lower_bound, add_none_lower.input.x),
        (ProcessWorkflow.input.align_regressor_upper_bound, add_none_upper.input.x),
        (ProcessWorkflow.input.reference_regressor_timeshift, add_none_lower.input.y),
        (ProcessWorkflow.input.reference_regressor_timeshift, add_none_upper.input.y),
        (add_none_lower.output.output, ProcessWorkflow.output.align_regressor_absolute_lower_bound),
        (add_none_upper.output.output, ProcessWorkflow.output.align_regressor_absolute_upper_bound),
        # iterate correlate
        (ProcessWorkflow.input.nr_parallel_processes, iterate_correlate_wf.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_correlate_wf.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_correlate_wf.input.boldIter_signal_timeseries_a),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_correlate_wf.input.signal_timeseries_b),
        (ProcessWorkflow.input.sample_time, iterate_correlate_wf.input.time_step),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_wf.input.bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_wf.input.window),
        (ProcessWorkflow.input.correlation_phat, iterate_correlate_wf.input.phat),
        (ProcessWorkflow.input.correlation_peak_threshold, iterate_correlate_wf.input.peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_correlate_wf.input.multi_peak_strategy),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_correlate_wf.input.ref_timeshift),
        (ValueNode(False).output.value, iterate_correlate_wf.input.boldIter_concat_array_outputs),
        (add_none_lower.output.output, iterate_correlate_wf.input.lower_limit),
        (add_none_upper.output.output, iterate_correlate_wf.input.upper_limit),
        (iterate_correlate_wf.output[("boldIter_timeshifts", "boldIter_correlations", "boldIter_fit_status")] , ProcessWorkflow.output._),
        # filter timeshift
        (ProcessWorkflow.input.timeseries_masker, filter_timeshifts.input.timeseries_masker),
        (ProcessWorkflow.input.filter_timeshifts_size, filter_timeshifts.input.size),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, filter_timeshifts.input.filter_type),
        (ProcessWorkflow.input.filter_timeshifts_smooth_fwhm, filter_timeshifts.input.smooth_fwhm),
        (iterate_correlate_wf.output.boldIter_timeshift_maxcorr, filter_timeshifts.input.timeshift_maxcorr),
        (iterate_correlate_wf.output.boldIter_maxcorr, filter_timeshifts.input.maxcorr),
        (iterate_correlate_wf.output.boldIter_timeshifts, filter_timeshifts.input.timeshifts),
        (iterate_correlate_wf.output.boldIter_correlations, filter_timeshifts.input.correlations),
        (filter_timeshifts.output.filtered_timeshift_maxcorr, ProcessWorkflow.output.boldIter_timeshift_maxcorr),
        (iterate_correlate_wf.output.boldIter_fit_status, filter_timeshifts.input.fit_status),
        (filter_timeshifts.output.filtered_maxcorr, ProcessWorkflow.output.boldIter_maxcorr),
        # histogram peak
        (filter_timeshifts.output.filtered_timeshift_maxcorr, hist_peak.input.values),
        (hist_peak.output.histogram_peak, ProcessWorkflow.output.histogram_timeshift_peak),
    ),
    description="find timeshift wf"
)
# find timeshift
refine_regressor_find_timeshift = find_timeshift_wf.copy(description="refine regressor - find timeshift")
# mask 
refine_regressor_mask_timeseries = MaskTimeSeries("refine regressor - mask timeseries")
refine_regressor_mask_timeshifts = MaskTimeSeries("refine regressor - mask timeshift")
# masked peak
masked_hist_peak = HistPeak(description="refine regressor masked histogram peak")
# align bold
refine_regressor_align_bold = IteratingNode(AlignTimeSeries(), iterating_inputs=("timeseries", "timeshift"), iterating_name="refine", description="refine regressor - align").setDefaultInputs(refineIter_nr_parallel_processes = -1)
# pca reduce timeseries
refine_regressor_pca_reduce = PCAReducedTimeSeries(description="refine regressor - pca reduce")

# %%
# refine regressor wf
refine_regressor_wf = ProcessWorkflow(
    (
        # find timeshift
        (ProcessWorkflow.input._, refine_regressor_find_timeshift.input.all / refine_regressor_find_timeshift.input[("show_pbar", "maxcorr_bipolar", "filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "timeseries_masker")]),
        (ValueNode(None).output.value, refine_regressor_find_timeshift.input[("filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "timeseries_masker")]),
        (ValueNode(False).output.value, refine_regressor_find_timeshift.input.maxcorr_bipolar),
        (ProcessWorkflow.input.show_pbar_refine, refine_regressor_find_timeshift.input.show_pbar),
        # mask timeseries
        (ProcessWorkflow.input.bold_signal_timeseries.T, refine_regressor_mask_timeseries.input.timeseries),
        (refine_regressor_find_timeshift.output.boldIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeseries.input.mask),
        # mask timeshift
        (refine_regressor_find_timeshift.output.boldIter_timeshift_maxcorr, refine_regressor_mask_timeshifts.input.timeseries),
        (refine_regressor_find_timeshift.output.boldIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeshifts.input.mask),
        # masked hist peak
        (refine_regressor_mask_timeshifts.output.masked_timeseries, masked_hist_peak.input.values),
        # align
        (ProcessWorkflow.input.sample_time, refine_regressor_align_bold.input.time_step),
        (ProcessWorkflow.input.nr_parallel_processes, refine_regressor_align_bold.input.refineIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar_refine, refine_regressor_align_bold.input.refineIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.shape[0], refine_regressor_align_bold.input.length),
        (ValueNode(False).output.value, refine_regressor_align_bold.input.refineIter_concat_array_outputs),
        (refine_regressor_mask_timeseries.output.masked_timeseries, refine_regressor_align_bold.input.refineIter_timeseries),
        (masked_hist_peak.output.histogram_peak - refine_regressor_mask_timeshifts.output.masked_timeseries, refine_regressor_align_bold.input.refineIter_timeshift),
        (ValueNode(False).output.value, refine_regressor_align_bold.input.fill_nan),
        # pca reduce
        (ProcessWorkflow.input.refine_regressor_explained_variance, refine_regressor_pca_reduce.input.explained_variance),
        (refine_regressor_align_bold.output.refineIter_aligned_timeseries.T, refine_regressor_pca_reduce.input.timeseries),
        (refine_regressor_pca_reduce.output.reduced_timeseries.mean(axis = 1), ProcessWorkflow.output.refined_regressor_signal_timeseries),
        # refined timeshift is zero since bold signals have been shifted to timeshift_peak
        (refine_regressor_find_timeshift.output.histogram_timeshift_peak - masked_hist_peak.output.histogram_peak, ProcessWorkflow.output.refined_reference_regressor_timeshift),
    ), description="refine regressor timeseries"
)
# %%
# recursive refine regressor
recursively_refine_regressor = RecursiveNode(refine_regressor_wf, recursive_map=(
    (refine_regressor_wf.input.regressor_signal_timeseries, refine_regressor_wf.output.refined_regressor_signal_timeseries),
    (refine_regressor_wf.input.reference_regressor_timeshift, refine_regressor_wf.output.refined_reference_regressor_timeshift),
), description="recursively refine regressor timeseries"
)

# %%
##############################################
# dynamic time warping
##############################################
dtw = DTW(description="dynamic time warping")
# reference target correlate
target_reference_correlate = Correlate(description="dtw target reference correlate")
# global regressor align
target_reference_align = AlignTimeSeries(description="dtw target reference regressor")
dynamic_time_warping_wf = ProcessWorkflow(
    (
        # target reference correlate
        (ProcessWorkflow.input.sample_time, target_reference_correlate.input.time_step),
        (ProcessWorkflow.input.target_align_reference_lower_bound, target_reference_correlate.input.lower_limit),
        (ProcessWorkflow.input.target_align_reference_upper_bound, target_reference_correlate.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, target_reference_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, target_reference_correlate.input.phat),
        (ValueNode(0.0).output.value, target_reference_correlate.input.peak_threshold),
        (ValueNode("max").output.value, target_reference_correlate.input.multi_peak_strategy),
        (ValueNode(None).output.value, target_reference_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, target_reference_correlate.input.bipolar),
        (ProcessWorkflow.input.target_timeseries , target_reference_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.reference_timeseries, target_reference_correlate.input.signal_timeseries_b),
        # target reference align
        (ProcessWorkflow.input.sample_time, target_reference_align.input.time_step),
        (ProcessWorkflow.input.target_timeseries.size, target_reference_align.input.length),
        (ProcessWorkflow.input.reference_timeseries, target_reference_align.input.timeseries),
        (target_reference_correlate.output.timeshift_maxcorr, target_reference_align.input.timeshift),
        (ValueNode(False).output.value, target_reference_align.input.fill_nan),
        # dtw
        (ProcessWorkflow.input.sample_time, dtw.input.time_step),
        (ProcessWorkflow.input.dispersion, dtw.input.window),
        (ProcessWorkflow.input.target_timeseries, dtw.input.target_timeseries),
        (target_reference_align.output.aligned_timeseries, dtw.input.reference_timeseries),
        (dtw.output.warped_timeseries, ProcessWorkflow.output.warped_timeseries),
    ), description="dynamic time warping wf"
)
# %%
no_dtw = ProcessWorkflow(
    (
        (ProcessWorkflow.input.target_timeseries, ProcessWorkflow.output.warped_timeseries),
    ), description="no dtw"
)
conditional_dtw = ConditionalNode("do_dtw", {True : dynamic_time_warping_wf, False : no_dtw}, False, description="conditional do dtw")

# %%
##############################################
# get regression confounds wf
##############################################

# # load motion confounds
# load_motion_confounds = MotionConfounds(description="get motion confounds")

# # threshold motion regressor maxcorr
# # motion_regressor_maxcorr = IteratingNode(Correlate(description="correlate regressor with motion confound"), ("timeseries_a", ), "motion", exclude_outputs=("timeshift_maxcorr", "timeshifts", "correlations"), show_pbar=False, description="compute maxcorr between regresssor and motion confounds")
# # get_confounds_below_thr = CustomNode(lambda motion_confounds_df, motion_regressor_maxcorr, confound_regressor_correlation_threshold = None : 
# #                                         motion_confounds_df.loc[:, np.abs(motion_regressor_maxcorr) < confound_regressor_correlation_threshold] \
# #                                             if confound_regressor_correlation_threshold is not None else motion_confounds_df, 
# #                                         outputs="thresholded_motion_confounds_df", description="threshold motion confounds"
# #                                     )

# downsample confounds
down_sample_confounds_df = DownsampleTimeSeries(description="down sample confounds df")

# get_regression_confounds_wf = ProcessWorkflow(
#     (
#         # motion confounds
#         (ProcessWorkflow.input.confounds_signal_df, load_motion_confounds.input.confounds_df),
#         (ProcessWorkflow.input.motion_derivatives, load_motion_confounds.input.derivatives),
#         (ProcessWorkflow.input.motion_powers, load_motion_confounds.input.powers),
#         (load_motion_confounds.output.motion_confound_names, ProcessWorkflow.output.motion_confound_names),
#         # # motion regressor maxcorr
#         # (ProcessWorkflow.input.regressor_signal_timeseries, motion_regressor_maxcorr.input.timeseries_b),
#         # (ProcessWorkflow.input.correlation_window, motion_regressor_maxcorr.input.window),
#         # (load_motion_confounds.output.motion_confounds_df.to_numpy().T, motion_regressor_maxcorr.input.motionIter_timeseries_a),
#         # (ValueNode(1).output.value, motion_regressor_maxcorr.input.time_step),
#         # (ValueNode(None).output.value, motion_regressor_maxcorr.input.lower_limit),
#         # (ValueNode(None).output.value, motion_regressor_maxcorr.input.upper_limit),
#         # (ValueNode(True).output.value, motion_regressor_maxcorr.input.bipolar),
#         # (motion_regressor_maxcorr.output.motionIter_maxcorr, ProcessWorkflow.output.motion_regressor_maxcorr),
#         # # get confounds below threshold
#         # (ProcessWorkflow.input.confound_regressor_correlation_threshold, get_confounds_below_thr.input.confound_regressor_correlation_threshold),
#         # (load_motion_confounds.output.motion_confounds_df, get_confounds_below_thr.input.motion_confounds_df),
#         # (motion_regressor_maxcorr.output.motionIter_maxcorr, get_confounds_below_thr.input.motion_regressor_maxcorr),
#         # down sample confounds
#         (ProcessWorkflow.input.down_sampling_factor, down_sample_confounds_df.input.down_sampling_factor),
#         (load_motion_confounds.output.motion_confounds_df, down_sample_confounds_df.input.timeseries),
#         (down_sample_confounds_df.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_regression_confounds_signal_df)
        
#     ), description="get regression confounds wf"
# )


# load motion confounds
pass_old_confounds = ProcessWorkflow(
    (
        (ProcessWorkflow.input.old_confounds_df, ProcessWorkflow.output.confounds_df),
    ))
cond_load_motion_confounds = ConditionalNode("include_motion_confounds", {True : MotionConfounds(description="get motion confounds"), False : pass_old_confounds}, default_condition=False)
cond_load_drift_confounds = ConditionalNode("include_drift_confounds", {True : DriftConfounds(description="get drift confounds"), False : pass_old_confounds}, default_condition=False)
cond_load_spike_confounds = ConditionalNode("include_spike_confounds", {True : SpikeConfounds(description="get spike confounds"), False : pass_old_confounds}, default_condition=False)
def addConstantConfound(confounds_df):
    if confounds_df is None:
        return None
    else:
        if "constant" not in confounds_df:
            confounds_df["constant"] = 1.0
        return confounds_df
add_constant_confound = CustomNode(addConstantConfound, ("confounds_df",), "add constant confound")

get_regression_confounds_wf = ProcessWorkflow(
    (
        # motion confounds
        (ProcessWorkflow.input.include_motion_confounds, cond_load_motion_confounds.input.include_motion_confounds),
        (ProcessWorkflow.input.confounds_signal_df, cond_load_motion_confounds.input.confounds_df),
        (ProcessWorkflow.input.motion_derivatives, cond_load_motion_confounds.input.derivatives),
        (ProcessWorkflow.input.motion_powers, cond_load_motion_confounds.input.powers),
        (ValueNode(None).output.value, cond_load_motion_confounds.input.old_confounds_df),
        # drift confounds
        (ProcessWorkflow.input._, cond_load_drift_confounds.input[("include_drift_confounds", "drift_high_pass", "drift_model", "drift_order")]),
        (ProcessWorkflow.input.bold_signal_timeseries.shape[0], cond_load_drift_confounds.input.nr_measurements),
        (ProcessWorkflow.input.time_step, cond_load_drift_confounds.input.tr),
        (cond_load_motion_confounds.output.confounds_df, cond_load_drift_confounds.input.old_confounds_df),
        (ValueNode(0.5).output.value, cond_load_drift_confounds.input.ref_slice),
        # spike confounds
        (ProcessWorkflow.input.bold_signal_timeseries, cond_load_spike_confounds.input.bold_data),
        (ProcessWorkflow.input.include_spike_confounds, cond_load_spike_confounds.input.include_spike_confounds),
        (ProcessWorkflow.input.spike_diff_cutoff, cond_load_spike_confounds.input.difference_cutoff),
        (ProcessWorkflow.input.spike_global_cutoff, cond_load_spike_confounds.input.global_cutoff),
        (cond_load_drift_confounds.output.confounds_df, cond_load_spike_confounds.input.old_confounds_df),
        # add constant confound
        (cond_load_spike_confounds.output.confounds_df, add_constant_confound.input.confounds_df),
        # down sample confounds
        (ProcessWorkflow.input.down_sampling_factor, down_sample_confounds_df.input.down_sampling_factor),
        (add_constant_confound.output.confounds_df, down_sample_confounds_df.input.timeseries),
        (down_sample_confounds_df.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_regression_confounds_signal_df)
    ), description="get regression confounds wf"
)

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
        (ProcessWorkflow.input.global_signal_timeseries, global_regressor_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.regressor_signal_timeseries, global_regressor_correlate.input.signal_timeseries_b),
        (ProcessWorkflow.input.sample_time, global_regressor_correlate.input.time_step),
        (ProcessWorkflow.input.correlation_window, global_regressor_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, global_regressor_correlate.input.phat),
        (ProcessWorkflow.input.correlation_peak_threshold, global_regressor_correlate.input.peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, global_regressor_correlate.input.multi_peak_strategy),
        (ValueNode(0).output.value, global_regressor_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_regressor_correlate.input.bipolar),
        (ProcessWorkflow.input.align_regressor_lower_bound, global_regressor_correlate.input.lower_limit),
        (ProcessWorkflow.input.align_regressor_upper_bound, global_regressor_correlate.input.upper_limit),
        (global_regressor_correlate.output.all, ProcessWorkflow.output._),
        # align downsample
        (ProcessWorkflow.input._, global_regressor_align_downsample.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.regressor_signal_timeseries, global_regressor_align_downsample.input.align_timeseries),
        (ProcessWorkflow.input.global_signal_timeseries, global_regressor_align_downsample.input.ref_timeseries),
        (global_regressor_correlate.output.timeshift_maxcorr, global_regressor_align_downsample.input.timeshift),
        (global_regressor_align_downsample.output.all / global_regressor_align_downsample.output["down_sampled_ref_timeseries", "down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (global_regressor_align_downsample.output.down_sampled_ref_timeseries, ProcessWorkflow.output.down_sampled_global_signal_timeseries),
        (global_regressor_align_downsample.output.down_sampled_aligned_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_regressor_signal_timeseries),
        # regress
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, global_regressor_regression.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_regressor_regression.input.confound_regressor_correlation_threshold),
        (global_regressor_align_downsample.output.down_sampled_ref_timeseries, global_regressor_regression.input.dv_ts),
        (global_regressor_align_downsample.output.down_sampled_aligned_timeseries, global_regressor_regression.input.regressor_timeseries),
        (global_regressor_regression.output.all / global_regressor_regression.output["design_matrix", "betas", "predictions"], ProcessWorkflow.output._),
    ),
    description="global regressor timeshift and beta wf"
)

# cond_global_regressor_regression = ConditionalNode("regressor", {True : global_regressor_regression_wf, False : None}, description="conditional regress regressor and regressor timseries")
# %%
setup_regression_wf = ProcessWorkflow(
    (
        # timeseries signal wf
        (ProcessWorkflow.input._, signal_timeseries_wf.input.all),
        (signal_timeseries_wf.output.all - signal_timeseries_wf.output.regressor_signal_timeseries, ProcessWorkflow.output._),
        # refine regressor 
        (ProcessWorkflow.input._, recursively_refine_regressor.input["sample_time", "align_regressor_lower_bound", "align_regressor_upper_bound", 'nr_parallel_processes', 'show_pbar', 'correlation_window', 'correlation_phat', 'correlation_peak_threshold', 'correlation_multi_peak_strategy', "refine_regressor_correlation_threshold", 'refine_regressor_explained_variance']),
        (ProcessWorkflow.input.refine_regressor_nr_recursions, recursively_refine_regressor.input.nr_recursions),
        (signal_timeseries_wf.output.bold_signal_timeseries, recursively_refine_regressor.input.bold_signal_timeseries),
        (signal_timeseries_wf.output.regressor_signal_timeseries, recursively_refine_regressor.input.Init_refined_regressor_signal_timeseries),
        (ValueNode(0).output.value, recursively_refine_regressor.input.Init_refined_reference_regressor_timeshift),
        (ValueNode(False).output.value, recursively_refine_regressor.input.show_pbar_refine),
        (recursively_refine_regressor.output.Rec_refined_reference_regressor_timeshift, ProcessWorkflow.output.setup_reference_regressor_timeshift),
        # dynamic time warping regressor to global
        (ProcessWorkflow.input._, conditional_dtw.input[('sample_time', 'correlation_phat', 'correlation_window', 'do_dtw')]),
        (ProcessWorkflow.input.align_regressor_lower_bound, conditional_dtw.input.target_align_reference_lower_bound),
        (ProcessWorkflow.input.align_regressor_upper_bound, conditional_dtw.input.target_align_reference_upper_bound),
        (ValueNode(10.0).output.value, conditional_dtw.input.dispersion),
        (signal_timeseries_wf.output.regressor_signal_timeseries, conditional_dtw.input.reference_timeseries),
        (recursively_refine_regressor.output.Rec_refined_regressor_signal_timeseries, conditional_dtw.input.target_timeseries),
        (conditional_dtw.output.warped_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        # signal characterics
        (ProcessWorkflow.input.correlation_window, signal_characterics_wf.input.correlation_window),
        (ProcessWorkflow.input.correlation_phat, signal_characterics_wf.input.correlation_phat),
        (ProcessWorkflow.input.sample_time, signal_characterics_wf.input.sample_time),
        (conditional_dtw.output.warped_timeseries, signal_characterics_wf.input.regressor_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, signal_characterics_wf.input.global_signal_timeseries),
        (signal_characterics_wf.output.all, ProcessWorkflow.output._),
        # get regression confounds wf
        (ProcessWorkflow.input.down_sampling_factor, get_regression_confounds_wf.input.down_sampling_factor),
        (ProcessWorkflow.input.sample_time, get_regression_confounds_wf.input.time_step),
        (ProcessWorkflow.input._, get_regression_confounds_wf.input["drift_high_pass", "drift_model", "drift_order", "include_drift_confounds", "include_motion_confounds", "include_spike_confounds", "motion_derivatives", "motion_powers", "spike_diff_cutoff", "spike_global_cutoff"]),
        (signal_timeseries_wf.output.bold_signal_timeseries, get_regression_confounds_wf.input.bold_signal_timeseries),
        (signal_timeseries_wf.output.confounds_signal_df, get_regression_confounds_wf.input.confounds_signal_df),
        (get_regression_confounds_wf.output.all, ProcessWorkflow.output._),
        # global_regressor_regression_wf
        (ProcessWorkflow.input._, global_regressor_regression_wf.input[("sample_time", "down_sampling_factor", "correlation_window", "correlation_phat", "correlation_peak_threshold", "correlation_multi_peak_strategy", "align_regressor_lower_bound", "align_regressor_upper_bound")]),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_regressor_regression_wf.input.confound_regressor_correlation_threshold),
        (signal_timeseries_wf.output.global_signal_timeseries, global_regressor_regression_wf.input.global_signal_timeseries),
        (conditional_dtw.output.warped_timeseries, global_regressor_regression_wf.input.regressor_signal_timeseries),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_signal_df, global_regressor_regression_wf.input.down_sampled_regression_confounds_signal_df),
        (global_regressor_regression_wf.output.timeshift_maxcorr, ProcessWorkflow.output.global_regressor_timeshift_maxcorr),
        (global_regressor_regression_wf.output.maxcorr, ProcessWorkflow.output.global_regressor_maxcorr),
        (global_regressor_regression_wf.output.timeshifts, ProcessWorkflow.output.global_regressor_timeshifts),
        (global_regressor_regression_wf.output.correlations, ProcessWorkflow.output.global_regressor_correlations),
        (global_regressor_regression_wf.output.down_sampled_global_signal_timeseries, ProcessWorkflow.output.down_sampled_global_signal_timeseries),
        (global_regressor_regression_wf.output.down_sampled_global_aligned_regressor_signal_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_regressor_signal_timeseries),
        (global_regressor_regression_wf.output.regressor_beta, ProcessWorkflow.output.global_regressor_beta),
        
    ),
    description="setup regression wf"
).setDefaultInputs(refine_regressor_correlation_threshold = 0.75, refine_regressor_nr_recursions = 0, refine_regressor_explained_variance = 0.8)


#%%
############################################################################################
# iterative cvr wf
############################################################################################

# %%
##############################################
# iterate align downsample over bold timeseries
##############################################
iterate_cvr_find_timeshift = find_timeshift_wf.copy(description="iterate cvr - find timeshift")
##############################################
# iterate align downsample over bold timeseries
##############################################
iterate_cvr_align_downsample = IteratingNode(align_downsample_wf.copy(), iterating_inputs=("ref_timeseries", "timeshift"), iterating_name="bold", description="iterate cvr – align and downsample").setDefaultInputs(boldIter_nr_parallel_processes = -1)

##############################################
# iterate calculate cvr over bold timeseries
##############################################
iterate_cvr_regress = IteratingNode(RegressCVR(), iterating_inputs=("dv_ts", "regressor_timeseries"), iterating_name="bold", exclude_outputs=("design_matrix", "betas"), description="iterate cvr – regress").setDefaultInputs(boldIter_nr_parallel_processes = -1)

# %%
iterate_cvr_wf = ProcessWorkflow(
    (
        # find timeshift 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_find_timeshift.input.nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_find_timeshift.input.show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries, iterate_cvr_find_timeshift.input.bold_signal_timeseries),
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
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_find_timeshift.input.regressor_signal_timeseries),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_cvr_find_timeshift.input.reference_regressor_timeshift),
        (iterate_cvr_find_timeshift.output.all / iterate_cvr_find_timeshift.output.histogram_timeshift_peak, ProcessWorkflow.output._),
        (iterate_cvr_find_timeshift.output.histogram_timeshift_peak, ProcessWorkflow.output.iterate_reference_regressor_timeshift),
        # iterate align downsample 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_align_downsample.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_align_downsample.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_cvr_align_downsample.input.boldIter_ref_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_cvr_align_downsample.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_cvr_align_downsample.input.down_sampling_factor),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_align_downsample.input.align_timeseries),
        (ValueNode(False).output.value, iterate_cvr_align_downsample.input.boldIter_concat_array_outputs),
        (iterate_cvr_find_timeshift.output.boldIter_timeshift_maxcorr, iterate_cvr_align_downsample.input.boldIter_timeshift),
        (iterate_cvr_align_downsample.output.all / iterate_cvr_align_downsample.output["boldIter_down_sampled_ref_timeseries", "boldIter_down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_ref_timeseries, ProcessWorkflow.output.boldIter_down_sampled_bold_signal_ts),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_aligned_timeseries, ProcessWorkflow.output.boldIter_down_sampled_aligned_regressor_signal_timeseries),
        # iterate calculate cvr
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_regress.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_regress.input.boldIter_show_pbar),
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, iterate_cvr_regress.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, iterate_cvr_regress.input.confound_regressor_correlation_threshold),
        (ValueNode(False).output.value, iterate_cvr_regress.input.boldIter_concat_array_outputs),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_ref_timeseries, iterate_cvr_regress.input.boldIter_dv_ts),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_aligned_timeseries, iterate_cvr_regress.input.boldIter_regressor_timeseries),
        (iterate_cvr_regress.output.all - iterate_cvr_regress.output.boldIter_predictions, ProcessWorkflow.output._),
        (iterate_cvr_regress.output.boldIter_predictions, ProcessWorkflow.output.boldIter_down_sampled_bold_signal_predictions)
    ),
    description="iterate cvr wf"
)

# %%

regression_wf = ProcessWorkflow(
    (
        # regression setup
        (ProcessWorkflow.input._, setup_regression_wf.input.all),
        (setup_regression_wf.output.all / setup_regression_wf.output.setup_reference_regressor_timeshift, ProcessWorkflow.output._),
        # iterative regression
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("sample_time", "timeseries_masker", "down_sampling_factor", "nr_parallel_processes", "show_pbar", "align_regressor_lower_bound", "align_regressor_upper_bound", "maxcorr_bipolar", "correlation_window", "correlation_phat", "correlation_peak_threshold", "correlation_multi_peak_strategy", "filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "confound_regressor_correlation_threshold")]),
        (setup_regression_wf.output.bold_signal_timeseries, iterate_cvr_wf.input.bold_signal_timeseries),
        (setup_regression_wf.output.regressor_signal_timeseries, iterate_cvr_wf.input.regressor_signal_timeseries),
        (setup_regression_wf.output.setup_reference_regressor_timeshift, iterate_cvr_wf.input.reference_regressor_timeshift),
        (setup_regression_wf.output.down_sampled_regression_confounds_signal_df, iterate_cvr_wf.input.down_sampled_regression_confounds_signal_df),
        (iterate_cvr_wf.output.all / iterate_cvr_wf.output.iterate_reference_regressor_timeshift, ProcessWorkflow.output._),
        (iterate_cvr_wf.output.iterate_reference_regressor_timeshift, ProcessWorkflow.output.reference_regressor_timeshift),
        # compute down sampled sample time
        (ProcessWorkflow.input.sample_time * ProcessWorkflow.input.down_sampling_factor, ProcessWorkflow.output.down_sampled_sample_time)
    ),
    description="regression wf"
)
# %%
