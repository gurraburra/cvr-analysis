# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.workflows.utils.signal_processing import DownsampleTimeSeries, MaskTimeSeries, DTW
from cvr_analysis.workflows.utils.data_computation import Correlate, AlignTimeSeries, RegressCVR, PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries, RMSTimeSeries, FilterTimeshifts, PCAReducedTimeSeries, HistPeak
from cvr_analysis.workflows.utils.confounds import MotionConfounds

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
doppler_baseline = BaselineTimeSeries(description="baseline doppler timeseries")
global_baseline = BaselineTimeSeries(description="baseline global timeseries")

# regressor
regressor_baseline = BaselineTimeSeries(description="baseline regressor timeseries")
regressor_signal_wf = ProcessWorkflow(
    (
        (ProcessWorkflow.input.regressor_timeseries, regressor_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, regressor_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, regressor_baseline.input.time_step),
        (ProcessWorkflow.input.regressor_timeseries - regressor_baseline.output.baseline, ProcessWorkflow.output.regressor_signal_timeseries),
    ),
    description="create regressor signal"
)
# get timeseries signal values
doppler_percentage = PercentageChangeTimeSeries(description="percentage doppler timeseries")
global_percentage = PercentageChangeTimeSeries(description="percentage global timeseries")


# %%
signal_timeseries_wf = ProcessWorkflow(
    (
        # doppler baseline
        (ProcessWorkflow.input.doppler_timeseries, doppler_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, doppler_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, doppler_baseline.input.time_step),
        # doppler percentage
        (ProcessWorkflow.input.doppler_timeseries, doppler_percentage.input.timeseries),
        (doppler_baseline.output.baseline, doppler_percentage.input.baseline),
        (doppler_percentage.output.percentage_timeseries, ProcessWorkflow.output.doppler_signal_timeseries),
        # global baseline
        (ProcessWorkflow.input.doppler_timeseries.mean(axis=1), global_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, global_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, global_baseline.input.time_step),
        # global percentage
        (ProcessWorkflow.input.doppler_timeseries.mean(axis=1), global_percentage.input.timeseries),
        (global_baseline.output.baseline, global_percentage.input.baseline),
        (global_percentage.output.percentage_timeseries, ProcessWorkflow.output.global_signal_timeseries),
        # regressor baseline
        (ProcessWorkflow.input.regressor_timeseries, regressor_signal_wf.input.regressor_timeseries),
        (ProcessWorkflow.input.baseline_strategy, regressor_signal_wf.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, regressor_signal_wf.input.sample_time),
        (regressor_signal_wf.output.regressor_signal_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
    ),
    description="convert timeseries into signals"
)

# %%
##############################################
# signal characteristics
##############################################
# power
regressor_rms = RMSTimeSeries(description="regressor rms")

# autocorrelation
regressor_autocorrelation = Correlate(description="regressor timeseries autocorrelation")

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
        ),
        description="signal characteristics"
)

##############################################
# refine global - regressor timeshift
##############################################
refined_global_regressor_correlate = Correlate(description="global regressor correlate")

# %%
setup_regression_wf = ProcessWorkflow(
    (
        # timeseries signal wf
        (ProcessWorkflow.input._, signal_timeseries_wf.input.all),
        (signal_timeseries_wf.output.all, ProcessWorkflow.output._),
        # signal characterics
        (ProcessWorkflow.input.correlation_window, signal_characterics_wf.input.correlation_window),
        (ProcessWorkflow.input.correlation_phat, signal_characterics_wf.input.correlation_phat),
        (ProcessWorkflow.input.sample_time, signal_characterics_wf.input.sample_time),
        (signal_timeseries_wf.output.regressor_signal_timeseries, signal_characterics_wf.input.regressor_signal_timeseries),
        (signal_characterics_wf.output.all, ProcessWorkflow.output._),
        # refine global align regressor 
        (ProcessWorkflow.input.sample_time, refined_global_regressor_correlate.input.time_step),
        (ProcessWorkflow.input.global_align_regressor_lower_bound, refined_global_regressor_correlate.input.lower_limit),
        (ProcessWorkflow.input.global_align_regressor_upper_bound, refined_global_regressor_correlate.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, refined_global_regressor_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, refined_global_regressor_correlate.input.phat),
        (ValueNode(0.0).output.value, refined_global_regressor_correlate.input.peak_threshold),
        (ValueNode("max").output.value, refined_global_regressor_correlate.input.multi_peak_strategy),
        (ValueNode(0.0).output.value, refined_global_regressor_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, refined_global_regressor_correlate.input.bipolar),
        (signal_timeseries_wf.output.global_signal_timeseries, refined_global_regressor_correlate.input.signal_timeseries_a),
        (signal_timeseries_wf.output.regressor_signal_timeseries, refined_global_regressor_correlate.input.signal_timeseries_b),
        (refined_global_regressor_correlate.output.timeshift_maxcorr, ProcessWorkflow.output.refined_global_regressor_timeshift),
    ),
    description="setup regression wf"
)
        
# %%
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


#%%
############################################################################################
# iterative cvr wf
############################################################################################

# %%
##############################################
# iterate align downsample over doppler timeseries
##############################################
iterate_cvr_find_timeshift = IteratingNode(Correlate(), iterating_inputs="signal_timeseries_a", iterating_name="doppler", description="iterate correlate doppler timeseries")
##############################################
# iterate align downsample over doppler timeseries
##############################################
iterate_cvr_align_downsample = IteratingNode(align_downsample_wf.copy(), iterating_inputs=("ref_timeseries", "timeshift"), iterating_name="doppler", description="iterate cvr – align and downsample").setDefaultInputs(dopplerIter_nr_parallel_processes = -1)

##############################################
# iterate calculate cvr over doppler timeseries
##############################################
iterate_cvr_regress = IteratingNode(RegressCVR(), iterating_inputs=("dv_ts", "regressor_timeseries"), iterating_name="doppler", exclude_outputs=("design_matrix", "betas"), description="iterate cvr – regress").setDefaultInputs(dopplerIter_nr_parallel_processes = -1)

##############################################
# compute absolute bounds
##############################################
add_none_lower = CustomNode(lambda x = None, y = None : x + y if x is not None and y is not None else None, description="add none")
add_none_upper = add_none_lower.copy()

# %%
iterate_cvr_wf = ProcessWorkflow(
    (
        # correlation bounds
        (ProcessWorkflow.input.align_regressor_lower_bound, add_none_lower.input.x),
        (ProcessWorkflow.input.align_regressor_upper_bound, add_none_upper.input.x),
        (ProcessWorkflow.input.refined_global_regressor_timeshift, add_none_lower.input.y),
        (ProcessWorkflow.input.refined_global_regressor_timeshift, add_none_upper.input.y),
        (add_none_lower.output.output, ProcessWorkflow.output.align_regressor_absolute_lower_bound),
        (add_none_upper.output.output, ProcessWorkflow.output.align_regressor_absolute_upper_bound),
        # find timeshift 
        (ValueNode(0).output.value, iterate_cvr_find_timeshift.input.dopplerIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_find_timeshift.input.dopplerIter_show_pbar),
        (ProcessWorkflow.input.doppler_signal_timeseries.T, iterate_cvr_find_timeshift.input.dopplerIter_signal_timeseries_a),
        (ProcessWorkflow.input.sample_time, iterate_cvr_find_timeshift.input.time_step),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_cvr_find_timeshift.input.bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_cvr_find_timeshift.input.window),
        (ProcessWorkflow.input.correlation_phat, iterate_cvr_find_timeshift.input.phat),
        (ProcessWorkflow.input.correlation_peak_threshold, iterate_cvr_find_timeshift.input.peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_cvr_find_timeshift.input.multi_peak_strategy),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_find_timeshift.input.signal_timeseries_b),
        (ProcessWorkflow.input.refined_global_regressor_timeshift, iterate_cvr_find_timeshift.input.ref_timeshift),
        (ValueNode(False).output.value, iterate_cvr_find_timeshift.input.dopplerIter_concat_array_outputs),
        (add_none_lower.output.output, iterate_cvr_find_timeshift.input.lower_limit),
        (add_none_upper.output.output, iterate_cvr_find_timeshift.input.upper_limit),
        (iterate_cvr_find_timeshift.output.all, ProcessWorkflow.output._),
        # iterate align downsample 
        (ValueNode(0).output.value, iterate_cvr_align_downsample.input.dopplerIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_align_downsample.input.dopplerIter_show_pbar),
        (ProcessWorkflow.input.doppler_signal_timeseries.T, iterate_cvr_align_downsample.input.dopplerIter_ref_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_cvr_align_downsample.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_cvr_align_downsample.input.down_sampling_factor),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_cvr_align_downsample.input.align_timeseries),
        (ValueNode(False).output.value, iterate_cvr_align_downsample.input.dopplerIter_concat_array_outputs),
        (iterate_cvr_find_timeshift.output.dopplerIter_timeshift_maxcorr, iterate_cvr_align_downsample.input.dopplerIter_timeshift),
        (iterate_cvr_align_downsample.output.all / iterate_cvr_align_downsample.output["dopplerIter_down_sampled_ref_timeseries", "dopplerIter_down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (iterate_cvr_align_downsample.output.dopplerIter_down_sampled_ref_timeseries, ProcessWorkflow.output.dopplerIter_down_sampled_doppler_signal_ts),
        (iterate_cvr_align_downsample.output.dopplerIter_down_sampled_aligned_timeseries, ProcessWorkflow.output.dopplerIter_down_sampled_aligned_regressor_signal_timeseries),
        # iterate calculate cvr
        (ValueNode(0).output.value, iterate_cvr_regress.input.dopplerIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_cvr_regress.input.dopplerIter_show_pbar),
        (ValueNode(None).output.value, iterate_cvr_regress.input.confounds_df),
        (ValueNode(None).output.value, iterate_cvr_regress.input.confound_regressor_correlation_threshold),
        (ValueNode(False).output.value, iterate_cvr_regress.input.dopplerIter_concat_array_outputs),
        (iterate_cvr_align_downsample.output.dopplerIter_down_sampled_ref_timeseries, iterate_cvr_regress.input.dopplerIter_dv_ts),
        (iterate_cvr_align_downsample.output.dopplerIter_down_sampled_aligned_timeseries, iterate_cvr_regress.input.dopplerIter_regressor_timeseries),
        (iterate_cvr_regress.output.all - iterate_cvr_regress.output.dopplerIter_predictions, ProcessWorkflow.output._),
        (iterate_cvr_regress.output.dopplerIter_predictions, ProcessWorkflow.output.dopplerIter_down_sampled_doppler_signal_predictions)
    ),
    description="iterate cvr wf"
)

# %%

regression_wf = ProcessWorkflow(
    (
        # regression setup
        (ProcessWorkflow.input._, setup_regression_wf.input.all),
        (setup_regression_wf.output.all, ProcessWorkflow.output._),
        # iterative regression
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("sample_time", "down_sampling_factor", "show_pbar", "align_regressor_lower_bound", "align_regressor_upper_bound", "maxcorr_bipolar", "correlation_window", "correlation_phat", "correlation_peak_threshold", "correlation_multi_peak_strategy")]),
        (setup_regression_wf.output.doppler_signal_timeseries, iterate_cvr_wf.input.doppler_signal_timeseries),
        (setup_regression_wf.output.regressor_signal_timeseries, iterate_cvr_wf.input.regressor_signal_timeseries),
        (setup_regression_wf.output.refined_global_regressor_timeshift, iterate_cvr_wf.input.refined_global_regressor_timeshift),
        (iterate_cvr_wf.output.all, ProcessWorkflow.output._),
        # compute down sampled sample time
        (ProcessWorkflow.input.sample_time * ProcessWorkflow.input.down_sampling_factor, ProcessWorkflow.output.down_sampled_sample_time)
    ),
    description="regression wf"
)
# %%
