# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.modalities_study.workflows.utils.signal_processing import DownsampleTimeSeries, MaskTimeSeries, DTW
from cvr_analysis.modalities_study.workflows.utils.data_computation import Correlate, AlignTimeSeries, RegressCVR, PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries, RMSTimeSeries, FilterTimeshifts, PCAReducedTimeSeries, HistPeak
from cvr_analysis.modalities_study.workflows.utils.confounds import MotionConfounds

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
co2_baseline = BaselineTimeSeries(description="baseline co2 timeseries")

# get timeseries signal values
bold_percentage = PercentageChangeTimeSeries(description="percentage bold timeseries")
global_percentage = PercentageChangeTimeSeries(description="percentage global timeseries")
confounds_std = StandardizeTimeSeries(description="standardize confounds")

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
        (ProcessWorkflow.input.bold_timeseries.mean(axis=1), global_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, global_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, global_baseline.input.time_step),
        # global percentage
        (ProcessWorkflow.input.bold_timeseries.mean(axis=1), global_percentage.input.timeseries),
        (global_baseline.output.baseline, global_percentage.input.baseline),
        (global_percentage.output.percentage_timeseries, ProcessWorkflow.output.global_signal_timeseries),
        # co2 baseline
        (ProcessWorkflow.input.co2_timeseries, co2_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, co2_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, co2_baseline.input.time_step),
        (ProcessWorkflow.input.co2_timeseries - co2_baseline.output.baseline, ProcessWorkflow.output.co2_signal_timeseries),
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
co2_rms = RMSTimeSeries(description="co2 rms")

# autocorrelation
regressor_autocorrelation = Correlate(description="regressor timeseries autocorrelation")
co2_autocorrelation = Correlate(description="co2 timeseries autocorrelation")

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
            (ProcessWorkflow.input.regressor_signal_timeseries, regressor_autocorrelation.input[("timeseries_a", "timeseries_b")]),
            (ValueNode(None).output.value, regressor_autocorrelation.input.multi_peak_strategy),
            (ValueNode(0).output.value, regressor_autocorrelation.input.ref_timeshift),
            (ValueNode(None).output.value, regressor_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, regressor_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, regressor_autocorrelation.input.bipolar),
            (regressor_autocorrelation.output.timeshifts, ProcessWorkflow.output.regressor_signal_autocorrelation_timeshifts),
            (regressor_autocorrelation.output.correlations, ProcessWorkflow.output.regressor_signal_autocorrelation_correlations),
            # co2 signal rms
            (ProcessWorkflow.input.co2_signal_timeseries, co2_rms.input.timeseries),
            (co2_rms.output.rms, ProcessWorkflow.output.co2_signal_rms),
            # co2 autocorrelation
            (ProcessWorkflow.input.correlation_window, co2_autocorrelation.input.window),
            (ProcessWorkflow.input.correlation_phat, co2_autocorrelation.input.phat),
            (ProcessWorkflow.input.sample_time, co2_autocorrelation.input.time_step),
            (ProcessWorkflow.input.co2_signal_timeseries, co2_autocorrelation.input[("timeseries_a", "timeseries_b")]),
            (ValueNode(None).output.value, co2_autocorrelation.input.multi_peak_strategy),
            (ValueNode(0).output.value, co2_autocorrelation.input.ref_timeshift),
            (ValueNode(None).output.value, co2_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, co2_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, co2_autocorrelation.input.bipolar),
            (co2_autocorrelation.output.timeshifts, ProcessWorkflow.output.co2_signal_autocorrelation_timeshifts),
            (co2_autocorrelation.output.correlations, ProcessWorkflow.output.co2_signal_autocorrelation_correlations),
        ),
        description="signal characteristics"
)
        
# %%

##############################################
# choose regressor
##############################################
regressor_dummy_wf = ProcessWorkflow(
    (
        (ProcessWorkflow.input.signal_timeseries, ProcessWorkflow.output.signal_timeseries),
    )
)
co2_dummy = regressor_dummy_wf.copy()
global_dummy = regressor_dummy_wf.copy()
dummy_input_mapping = {
                    "co2_signal_timeseries"     : co2_dummy.input.signal_timeseries, 
                    "global_signal_timeseries"  : global_dummy.input.signal_timeseries, 
                }
choose_regressor = ConditionalNode("use_co2_regressor", {True : co2_dummy, False : global_dummy}, default_condition=True, input_mapping = dummy_input_mapping, description="choose regressor")

##############################################
# refine regressor
##############################################
##--##--##--##--##--##--##--##--##--##--##--##
# find timeshift
##--##--##--##--##--##--##--##--##--##--##--##
# iterate correlate over bold timeseries
iterate_correlate_wf = IteratingNode(Correlate(), iterating_inputs="timeseries_a", iterating_name="bold", description="iterate correlate bold timeseries").setDefaultInputs(boldIter_nr_parallel_processes = -1)
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
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_correlate_wf.input.boldIter_timeseries_a),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_correlate_wf.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, iterate_correlate_wf.input.time_step),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_wf.input.bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_wf.input.window),
        (ProcessWorkflow.input.correlation_phat, iterate_correlate_wf.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_correlate_wf.input.multi_peak_strategy),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_correlate_wf.input.ref_timeshift),
        (add_none_lower.output.output, iterate_correlate_wf.input.lower_limit),
        (add_none_upper.output.output, iterate_correlate_wf.input.upper_limit),
        (iterate_correlate_wf.output.all - iterate_correlate_wf.output.boldIter_timeshift_maxcorr, ProcessWorkflow.output._),
        # filter timeshift
        (ProcessWorkflow.input.timeseries_masker, filter_timeshifts.input.timeseries_masker),
        (ProcessWorkflow.input.filter_timeshifts_maxcorr_threshold, filter_timeshifts.input.maxcorr_threshold),
        (ProcessWorkflow.input.filter_timeshifts_size, filter_timeshifts.input.size),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, filter_timeshifts.input.filter_type),
        (iterate_correlate_wf.output.boldIter_timeshift_maxcorr, filter_timeshifts.input.timeshifts),
        (iterate_correlate_wf.output.boldIter_maxcorr, filter_timeshifts.input.maxcorrs),
        (filter_timeshifts.output.filtered_timeshifts, ProcessWorkflow.output.boldIter_timeshift_maxcorr),
        # histogram peak
        (filter_timeshifts.output.filtered_timeshifts, hist_peak.input.values),
        (hist_peak.output.histogram_peak, ProcessWorkflow.output.refined_reference_regressor_timeshift),
    ),
    description="find timeshift wf"
).setDefaultInputs(filter_timeshifts_maxcorr_threshold = 0.75)
# find timeshift
refine_regressor_find_timeshift = find_timeshift_wf.copy(description="refine regressor - find timeshift")
# mask 
refine_regressor_mask_timeseries = MaskTimeSeries("refine regressor - mask timeseries")
refine_regressor_mask_timeshifts = MaskTimeSeries("refine regressor - mask timeshift")
# align bold
refine_regressor_align_bold = IteratingNode(AlignTimeSeries(), iterating_inputs=("timeseries", "timeshift"), iterating_name="refine", description="refine regressor - align").setDefaultInputs(refineIter_nr_parallel_processes = -1)
# pca reduce timeseries
refine_regressor_pca_reduce = PCAReducedTimeSeries(description="refine regressor - pca reduce")

# %%
# refine regressor wf
refine_regressor_wf = ProcessWorkflow(
    (
        # find timeshift
        (ProcessWorkflow.input._, refine_regressor_find_timeshift.input.all),
        # mask timeseries
        (ProcessWorkflow.input.bold_signal_timeseries.T, refine_regressor_mask_timeseries.input.timeseries),
        (refine_regressor_find_timeshift.output.boldIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeseries.input.mask),
        # mask timeshift
        (refine_regressor_find_timeshift.output.boldIter_timeshift_maxcorr, refine_regressor_mask_timeshifts.input.timeseries),
        (refine_regressor_find_timeshift.output.boldIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeshifts.input.mask),
        # align
        (ProcessWorkflow.input.sample_time, refine_regressor_align_bold.input.time_step),
        (ProcessWorkflow.input.nr_parallel_processes, refine_regressor_align_bold.input.refineIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, refine_regressor_align_bold.input.refineIter_show_pbar),
        (ProcessWorkflow.input.regressor_signal_timeseries.size, refine_regressor_align_bold.input.length),
        (refine_regressor_mask_timeseries.output.masked_timeseries, refine_regressor_align_bold.input.refineIter_timeseries),
        (refine_regressor_find_timeshift.output.refined_reference_regressor_timeshift - refine_regressor_mask_timeshifts.output.masked_timeseries, refine_regressor_align_bold.input.refineIter_timeshift),
        (ValueNode(True).output.value, refine_regressor_align_bold.input.fill_nan),
        # pca reduce
        (ProcessWorkflow.input.refine_regressor_explained_variance, refine_regressor_pca_reduce.input.explained_variance),
        (refine_regressor_align_bold.output.refineIter_aligned_timeseries.T, refine_regressor_pca_reduce.input.timeseries),
        (refine_regressor_pca_reduce.output.reduced_timeseries.mean(axis = 1), ProcessWorkflow.output.refined_regressor_signal_timeseries),
        # refined timeshift is zero since bold signals have been shifted to timeshift_peak
        (ValueNode(0.0).output.value, ProcessWorkflow.output.refined_reference_regressor_timeshift),
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
dynamic_time_warping = DTW(description="dynamic time warping")
no_dtw = ProcessWorkflow(((ProcessWorkflow.input.target_timeseries, ProcessWorkflow.output.warped_timeseries),), description="no dtw")
def maxLimit(lower_limit, upper_limit):
    if lower_limit is None and upper_limit is None:
        return None
    elif lower_limit is None:
        return abs(upper_limit)
    elif upper_limit is None:
        return abs(lower_limit)
    else:
        return max(abs(lower_limit), abs(upper_limit)),
max_limit = CustomNode(maxLimit, outputs=("max_limit",), description="get maximum of two limits")
conditional_dtw = ConditionalNode("do_dtw", {True : dynamic_time_warping, False : no_dtw}, False, description="conditional do dtw")

#%%
from process_control import ProcessWorkflow, _and_, _or_, ValueNode, not_
# run dwi or not
do_dtw = ProcessWorkflow(
    (
        (ProcessWorkflow.input.ensure_co2_units *_and_* (not_*ProcessWorkflow.input.use_co2_regressor *_or_* ProcessWorkflow.input.refine_regressor_nr_recursions > ValueNode(0).output.value), ProcessWorkflow.output.do_dtw),
    ), description="do dtw"
)
# %%
##############################################
# get regression confounds wf
##############################################

# load motion confounds
load_motion_confounds = MotionConfounds(description="get motion confounds")

# threshold motion regressor maxcorr
# motion_regressor_maxcorr = IteratingNode(Correlate(description="correlate regressor with motion confound"), ("timeseries_a", ), "motion", exclude_outputs=("timeshift_maxcorr", "timeshifts", "correlations"), show_pbar=False, description="compute maxcorr between regresssor and motion confounds")
# get_confounds_below_thr = CustomNode(lambda motion_confounds_df, motion_regressor_maxcorr, confound_regressor_correlation_threshold = None : 
#                                         motion_confounds_df.loc[:, np.abs(motion_regressor_maxcorr) < confound_regressor_correlation_threshold] \
#                                             if confound_regressor_correlation_threshold is not None else motion_confounds_df, 
#                                         outputs="thresholded_motion_confounds_df", description="threshold motion confounds"
#                                     )

# downsample confounds
down_sample_confounds_df = DownsampleTimeSeries(description="down sample confounds df")

get_regression_confounds_wf = ProcessWorkflow(
    (
        # motion confounds
        (ProcessWorkflow.input.confounds_signal_df, load_motion_confounds.input.confounds_df),
        (ProcessWorkflow.input.motion_derivatives, load_motion_confounds.input.derivatives),
        (ProcessWorkflow.input.motion_powers, load_motion_confounds.input.powers),
        (load_motion_confounds.output.motion_confounds_df.columns, ProcessWorkflow.output.motion_confound_names),
        # # motion regressor maxcorr
        # (ProcessWorkflow.input.regressor_signal_timeseries, motion_regressor_maxcorr.input.timeseries_b),
        # (ProcessWorkflow.input.correlation_window, motion_regressor_maxcorr.input.window),
        # (load_motion_confounds.output.motion_confounds_df.to_numpy().T, motion_regressor_maxcorr.input.motionIter_timeseries_a),
        # (ValueNode(1).output.value, motion_regressor_maxcorr.input.time_step),
        # (ValueNode(None).output.value, motion_regressor_maxcorr.input.lower_limit),
        # (ValueNode(None).output.value, motion_regressor_maxcorr.input.upper_limit),
        # (ValueNode(True).output.value, motion_regressor_maxcorr.input.bipolar),
        # (motion_regressor_maxcorr.output.motionIter_maxcorr, ProcessWorkflow.output.motion_regressor_maxcorr),
        # # get confounds below threshold
        # (ProcessWorkflow.input.confound_regressor_correlation_threshold, get_confounds_below_thr.input.confound_regressor_correlation_threshold),
        # (load_motion_confounds.output.motion_confounds_df, get_confounds_below_thr.input.motion_confounds_df),
        # (motion_regressor_maxcorr.output.motionIter_maxcorr, get_confounds_below_thr.input.motion_regressor_maxcorr),
        # down sample confounds
        (ProcessWorkflow.input.down_sampling_factor, down_sample_confounds_df.input.down_sampling_factor),
        (load_motion_confounds.output.motion_confounds_df, down_sample_confounds_df.input.timeseries),
        (down_sample_confounds_df.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_regression_confounds_signal_df)
        
    ), description="get regression confounds wf"
)

# %%
##############################################
# regressor co2 timeshift and beta
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
# regress bold and regressor
regressor_co2_correlate = Correlate(description="regressor co2 regress")
regressor_co2_align_downsample = align_downsample_wf.copy(description = "regressor co2 align downsample")
regressor_co2_regression = RegressCVR(description="regressor co2 regression")

regressor_co2_regression_wf = ProcessWorkflow(
    (
        # correlate 
        (ProcessWorkflow.input.regressor_signal_timeseries, regressor_co2_correlate.input.timeseries_a),
        (ProcessWorkflow.input.co2_signal_timeseries, regressor_co2_correlate.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, regressor_co2_correlate.input.time_step),
        (ProcessWorkflow.input.correlation_window, regressor_co2_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, regressor_co2_correlate.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, regressor_co2_correlate.input.multi_peak_strategy),
        (ValueNode(0).output.value, regressor_co2_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, regressor_co2_correlate.input.bipolar),
        (ValueNode(None).output.value, regressor_co2_correlate.input.lower_limit),
        (ValueNode(None).output.value, regressor_co2_correlate.input.upper_limit),
        (regressor_co2_correlate.output.all, ProcessWorkflow.output._),
        # align downsample
        (ProcessWorkflow.input._, regressor_co2_align_downsample.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.co2_signal_timeseries, regressor_co2_align_downsample.input.align_timeseries),
        (ProcessWorkflow.input.regressor_signal_timeseries, regressor_co2_align_downsample.input.ref_timeseries),
        (regressor_co2_correlate.output.timeshift_maxcorr, regressor_co2_align_downsample.input.timeshift),
        (regressor_co2_align_downsample.output.all / regressor_co2_align_downsample.output["down_sampled_ref_timeseries", "down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (regressor_co2_align_downsample.output.down_sampled_ref_timeseries, ProcessWorkflow.output.down_sampled_regressor_signal_timeseries),
        (regressor_co2_align_downsample.output.down_sampled_aligned_timeseries, ProcessWorkflow.output.down_sampled_regressor_aligned_co2_signal_timeseries),
        # regress
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, regressor_co2_regression.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, regressor_co2_regression.input.confound_regressor_correlation_threshold),
        (regressor_co2_align_downsample.output.down_sampled_ref_timeseries, regressor_co2_regression.input.bold_ts),
        (regressor_co2_align_downsample.output.down_sampled_aligned_timeseries, regressor_co2_regression.input.regressor_timeseries),
        (regressor_co2_regression.output.all / regressor_co2_regression.output["design_matrix", "betas", "predictions"], ProcessWorkflow.output._),
    ),
    description="regressor co2 timeshift and beta wf"
)


##############################################
# correlation bounds
##############################################
# global regressor correlate
correlate_global_regressor_timeseries = Correlate(description="global regressor timeshift")


# %%
setup_regression_wf = ProcessWorkflow(
    (
        # timeseries signal wf
        (ProcessWorkflow.input._, signal_timeseries_wf.input.all),
        (signal_timeseries_wf.output.all, ProcessWorkflow.output._),
        # choose regressor
        (ProcessWorkflow.input._, choose_regressor.input.use_co2_regressor),
        (signal_timeseries_wf.output.co2_signal_timeseries, choose_regressor.input.co2_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, choose_regressor.input.global_signal_timeseries),
        # initial global regressor alignment
        (ProcessWorkflow.input.sample_time, correlate_global_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.initial_global_align_lower_bound, correlate_global_regressor_timeseries.input.lower_limit),
        (ProcessWorkflow.input.initial_global_align_upper_bound, correlate_global_regressor_timeseries.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, correlate_global_regressor_timeseries.input.window),
        (ProcessWorkflow.input.correlation_phat, correlate_global_regressor_timeseries.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, correlate_global_regressor_timeseries.input.multi_peak_strategy),
        (ValueNode(0).output.value, correlate_global_regressor_timeseries.input.ref_timeshift),
        (ValueNode(False).output.value, correlate_global_regressor_timeseries.input.bipolar),
        (signal_timeseries_wf.output.global_signal_timeseries, correlate_global_regressor_timeseries.input.timeseries_a),
        (choose_regressor.output.signal_timeseries, correlate_global_regressor_timeseries.input.timeseries_b),
        # refine regressor 
        (ProcessWorkflow.input._, recursively_refine_regressor.input["sample_time", "timeseries_masker", "align_regressor_lower_bound", "align_regressor_upper_bound", 'nr_parallel_processes', 'show_pbar', 'maxcorr_bipolar', 'correlation_window', 'correlation_phat', 'correlation_multi_peak_strategy', 'filter_timeshifts_maxcorr_threshold', 'filter_timeshifts_size', 'filter_timeshifts_filter_type', "refine_regressor_correlation_threshold", 'refine_regressor_explained_variance']),
        (ProcessWorkflow.input.refine_regressor_nr_recursions, recursively_refine_regressor.input.nr_recursions),
        (signal_timeseries_wf.output.bold_signal_timeseries, recursively_refine_regressor.input.bold_signal_timeseries),
        (choose_regressor.output.signal_timeseries, recursively_refine_regressor.input.Init_refined_regressor_signal_timeseries),
        (correlate_global_regressor_timeseries.output.timeshift_maxcorr, recursively_refine_regressor.input.Init_refined_reference_regressor_timeshift),
        (recursively_refine_regressor.output.Rec_refined_reference_regressor_timeshift, ProcessWorkflow.output.setup_reference_regressor_timeshift),
        # do dtw
        (ProcessWorkflow.input.ensure_co2_units, do_dtw.input.ensure_co2_units),
        (ProcessWorkflow.input.use_co2_regressor, do_dtw.input.use_co2_regressor),
        (ProcessWorkflow.input.refine_regressor_nr_recursions, do_dtw.input.refine_regressor_nr_recursions),
        (do_dtw.output.do_dtw, ProcessWorkflow.output.do_dtw),
        # dynamic time warping
        (ProcessWorkflow.input.sample_time, conditional_dtw.input.time_step),
        (ProcessWorkflow.input.initial_global_align_lower_bound, max_limit.input.lower_limit),
        (ProcessWorkflow.input.initial_global_align_upper_bound, max_limit.input.upper_limit),
        (do_dtw.output.do_dtw, conditional_dtw.input.do_dtw),
        (max_limit.output.max_limit, conditional_dtw.input.window),
        (choose_regressor.output.signal_timeseries, conditional_dtw.input.target_timeseries),
        (recursively_refine_regressor.output.Rec_refined_regressor_signal_timeseries, conditional_dtw.input.reference_timeseries),
        (conditional_dtw.output.warped_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        # signal characterics
        (ProcessWorkflow.input.correlation_window, signal_characterics_wf.input.correlation_window),
        (ProcessWorkflow.input.correlation_phat, signal_characterics_wf.input.correlation_phat),
        (ProcessWorkflow.input.sample_time, signal_characterics_wf.input.sample_time),
        (recursively_refine_regressor.output.Rec_refined_regressor_signal_timeseries, signal_characterics_wf.input.regressor_signal_timeseries),
        (signal_timeseries_wf.output.co2_signal_timeseries, signal_characterics_wf.input.co2_signal_timeseries),
        (signal_characterics_wf.output.all, ProcessWorkflow.output._),
        # get regression confounds wf
        (ProcessWorkflow.input.down_sampling_factor, get_regression_confounds_wf.input.down_sampling_factor),
        (signal_timeseries_wf.output.confounds_signal_df, get_regression_confounds_wf.input.confounds_signal_df),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_derivatives),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_powers),
        (get_regression_confounds_wf.output.all, ProcessWorkflow.output._),
        # regressor_co2_regression_wf
        (ProcessWorkflow.input._, regressor_co2_regression_wf.input[("sample_time", "down_sampling_factor", "correlation_window", "correlation_phat", "correlation_multi_peak_strategy")]),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, regressor_co2_regression_wf.input.confound_regressor_correlation_threshold),
        (signal_timeseries_wf.output.co2_signal_timeseries, regressor_co2_regression_wf.input.co2_signal_timeseries),
        (recursively_refine_regressor.output.Rec_refined_regressor_signal_timeseries, regressor_co2_regression_wf.input.regressor_signal_timeseries),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_signal_df, regressor_co2_regression_wf.input.down_sampled_regression_confounds_signal_df),
        (regressor_co2_regression_wf.output.timeshift_maxcorr, ProcessWorkflow.output.regressor_co2_timeshift_maxcorr),
        (regressor_co2_regression_wf.output.maxcorr, ProcessWorkflow.output.regressor_co2_maxcorr),
        (regressor_co2_regression_wf.output.timeshifts, ProcessWorkflow.output.regressor_co2_timeshifts),
        (regressor_co2_regression_wf.output.correlations, ProcessWorkflow.output.regressor_co2_correlations),
        (regressor_co2_regression_wf.output.down_sampled_regressor_signal_timeseries, ProcessWorkflow.output.down_sampled_regressor_signal_timeseries),
        (regressor_co2_regression_wf.output.down_sampled_regressor_aligned_co2_signal_timeseries, ProcessWorkflow.output.down_sampled_regressor_aligned_co2_signal_timeseries),
        (regressor_co2_regression_wf.output.regressor_beta, ProcessWorkflow.output.regressor_co2_beta),
        
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
iterate_cvr_align_downsample = IteratingNode(align_downsample_wf.copy(), iterating_inputs=("ref_timeseries", "timeshift"), iterating_name="bold", description="iterate align, downsample bold timeseries").setDefaultInputs(boldIter_nr_parallel_processes = -1)

##############################################
# iterate calculate cvr over bold timeseries
##############################################
iterate_cvr_regress = IteratingNode(RegressCVR(), iterating_inputs=("bold_ts", "regressor_timeseries"), iterating_name="bold", exclude_outputs=("design_matrix", "betas"), description="iterative calculate cvr").setDefaultInputs(boldIter_nr_parallel_processes = -1)

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
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_cvr_find_timeshift.input.correlation_multi_peak_strategy),
        (ProcessWorkflow.input.filter_timeshifts_maxcorr_threshold, iterate_cvr_find_timeshift.input.filter_timeshifts_maxcorr_threshold),
        (ProcessWorkflow.input.filter_timeshifts_size, iterate_cvr_find_timeshift.input.filter_timeshifts_size),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, iterate_cvr_find_timeshift.input.filter_timeshifts_filter_type),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_find_timeshift.input.regressor_signal_timeseries),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_cvr_find_timeshift.input.reference_regressor_timeshift),
        (iterate_cvr_find_timeshift.output.all / iterate_cvr_find_timeshift.output.refined_reference_regressor_timeshift, ProcessWorkflow.output._),
        (iterate_cvr_find_timeshift.output.refined_reference_regressor_timeshift, ProcessWorkflow.output.iterate_reference_regressor_timeshift),
        # iterate align downsample 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_align_downsample.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_align_downsample.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_cvr_align_downsample.input.boldIter_ref_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_cvr_align_downsample.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_cvr_align_downsample.input.down_sampling_factor),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_align_downsample.input.align_timeseries),
        (iterate_cvr_find_timeshift.output.boldIter_timeshift_maxcorr, iterate_cvr_align_downsample.input.boldIter_timeshift),
        (iterate_cvr_align_downsample.output.all / iterate_cvr_align_downsample.output["boldIter_down_sampled_ref_timeseries", "boldIter_down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_ref_timeseries, ProcessWorkflow.output.boldIter_down_sampled_bold_signal_ts),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_aligned_timeseries, ProcessWorkflow.output.boldIter_down_sampled_aligned_regressor_signal_timeseries),
        # iterate calculate cvr
        (ProcessWorkflow.input.nr_parallel_processes, iterate_cvr_regress.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_regress.input.boldIter_show_pbar),
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, iterate_cvr_regress.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, iterate_cvr_regress.input.confound_regressor_correlation_threshold),
        (iterate_cvr_align_downsample.output.boldIter_down_sampled_ref_timeseries, iterate_cvr_regress.input.boldIter_bold_ts),
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
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("sample_time", "timeseries_masker", "down_sampling_factor", "nr_parallel_processes", "show_pbar", "align_regressor_lower_bound", "align_regressor_upper_bound", "maxcorr_bipolar", "correlation_window", "correlation_phat", "correlation_multi_peak_strategy", "filter_timeshifts_maxcorr_threshold", "filter_timeshifts_size", "filter_timeshifts_filter_type", "confound_regressor_correlation_threshold")]),
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
