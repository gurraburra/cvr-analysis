# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.modalities_study.workflows.utils.signal_processing import DownsampleTimeSeries
from cvr_analysis.modalities_study.workflows.utils.data_computation import Correlate, AlignTimeSeries, RegressCVR, PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries, RMSTimeSeries, FilterTimeshifts
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
global_rms = RMSTimeSeries(description="global rms")
co2_rms = RMSTimeSeries(description="co2 rms")

# autocorrelation
global_autocorrelation = Correlate(description="global timeseries autocorrelation")
co2_autocorrelation = Correlate(description="co2 timeseries autocorrelation")

# %%
signal_characterics_wf = ProcessWorkflow(
        (
            # global signal power
            (ProcessWorkflow.input.global_signal_timeseries, global_rms.input.timeseries),
            (global_rms.output.rms, ProcessWorkflow.output.global_signal_rms),
            # global autocorrelation
            (ProcessWorkflow.input.correlation_window, global_autocorrelation.input.window),
            (ProcessWorkflow.input.correlation_phat, global_autocorrelation.input.phat),
            (ProcessWorkflow.input.sample_time, global_autocorrelation.input.time_step),
            (ProcessWorkflow.input.global_signal_timeseries, global_autocorrelation.input[("timeseries_a", "timeseries_b")]),
            (ValueNode(None).output.value, global_autocorrelation.input.multi_peak_strategy),
            (ValueNode(0).output.value, global_autocorrelation.input.ref_timeshift),
            (ValueNode(None).output.value, global_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, global_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, global_autocorrelation.input.bipolar),
            (global_autocorrelation.output.timeshifts, ProcessWorkflow.output.global_signal_autocorrelation_timeshifts),
            (global_autocorrelation.output.correlations, ProcessWorkflow.output.global_signal_autocorrelation_correlations),
            # global signal rms
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
        description="singal characteristics"
)

# global autocorrelation
        
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
# regressor autocorrelation
##############################################


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
# global co2 timeshift and beta
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
##--##--##--##--##--##--##--##--##--##--##--##
# calculate cvr
##--##--##--##--##--##--##--##--##--##--##--##
# regress bold and regressor
global_co2_correlate = Correlate(description="global co2 regress")
global_co2_align_downsample = align_downsample_wf.copy(description = "global co2 align downsample")
global_co2_regression = RegressCVR(description="global co2 regression")

global_co2_regression_wf = ProcessWorkflow(
    (
        # correlate 
        (ProcessWorkflow.input.global_signal_timeseries, global_co2_correlate.input.timeseries_a),
        (ProcessWorkflow.input.co2_signal_timeseries, global_co2_correlate.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, global_co2_correlate.input.time_step),
        (ProcessWorkflow.input.correlation_window, global_co2_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, global_co2_correlate.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, global_co2_correlate.input.multi_peak_strategy),
        (ValueNode(0).output.value, global_co2_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, global_co2_correlate.input.bipolar),
        (ValueNode(None).output.value, global_co2_correlate.input.lower_limit),
        (ValueNode(None).output.value, global_co2_correlate.input.upper_limit),
        (global_co2_correlate.output.all, ProcessWorkflow.output._),
        # align downsample
        (ProcessWorkflow.input._, global_co2_align_downsample.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.co2_signal_timeseries, global_co2_align_downsample.input.align_timeseries),
        (ProcessWorkflow.input.global_signal_timeseries, global_co2_align_downsample.input.ref_timeseries),
        (global_co2_correlate.output.timeshift_maxcorr, global_co2_align_downsample.input.timeshift),
        (global_co2_align_downsample.output.all / global_co2_align_downsample.output["down_sampled_ref_timeseries", "down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (global_co2_align_downsample.output.down_sampled_ref_timeseries, ProcessWorkflow.output.down_sampled_global_signal_timeseries),
        (global_co2_align_downsample.output.down_sampled_aligned_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_co2_signal_timeseries),
        # regress
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, global_co2_regression.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_co2_regression.input.confound_regressor_correlation_threshold),
        (global_co2_align_downsample.output.down_sampled_ref_timeseries, global_co2_regression.input.bold_ts),
        (global_co2_align_downsample.output.down_sampled_aligned_timeseries, global_co2_regression.input.regressor_timeseries),
        (global_co2_regression.output.all / global_co2_regression.output["design_matrix", "betas", "predictions"], ProcessWorkflow.output._),
    ),
    description="global co2 timeshift and beta wf"
)


##############################################
# correlation bounds
##############################################
# global regressor correlate
correlate_global_regressor_timeseries = Correlate(description="global regressor timeshift")
# add if not None
add_none_lower = CustomNode(lambda x = None, y = None : x + y if x is not None and y is not None else None, description="add none")
add_none_upper = add_none_lower.copy()

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
        (signal_timeseries_wf.output.global_signal_timeseries, signal_characterics_wf.input.global_signal_timeseries),
        (signal_timeseries_wf.output.co2_signal_timeseries, signal_characterics_wf.input.co2_signal_timeseries),
        (signal_characterics_wf.output.all, ProcessWorkflow.output._),
        # choose regressor
        (ProcessWorkflow.input._, choose_regressor.input.use_co2_regressor),
        (signal_timeseries_wf.output.co2_signal_timeseries, choose_regressor.input.co2_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, choose_regressor.input.global_signal_timeseries),
        (choose_regressor.output.signal_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        # get regression confounds wf
        (ProcessWorkflow.input.down_sampling_factor, get_regression_confounds_wf.input.down_sampling_factor),
        (signal_timeseries_wf.output.confounds_signal_df, get_regression_confounds_wf.input.confounds_signal_df),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_derivatives),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_powers),
        (get_regression_confounds_wf.output.all, ProcessWorkflow.output._),
        # global_co2_regression_wf
        (ProcessWorkflow.input._, global_co2_regression_wf.input[("sample_time", "down_sampling_factor", "correlation_window", "correlation_phat", "correlation_multi_peak_strategy")]),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_co2_regression_wf.input.confound_regressor_correlation_threshold),
        (signal_timeseries_wf.output.co2_signal_timeseries, global_co2_regression_wf.input.co2_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, global_co2_regression_wf.input.global_signal_timeseries),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_signal_df, global_co2_regression_wf.input.down_sampled_regression_confounds_signal_df),
        (global_co2_regression_wf.output.timeshift_maxcorr, ProcessWorkflow.output.global_co2_timeshift_maxcorr),
        (global_co2_regression_wf.output.maxcorr, ProcessWorkflow.output.global_co2_maxcorr),
        (global_co2_regression_wf.output.timeshifts, ProcessWorkflow.output.global_co2_timeshifts),
        (global_co2_regression_wf.output.correlations, ProcessWorkflow.output.global_co2_correlations),
        (global_co2_regression_wf.output.down_sampled_global_signal_timeseries, ProcessWorkflow.output.down_sampled_global_signal_timeseries),
        (global_co2_regression_wf.output.down_sampled_global_aligned_co2_signal_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_co2_signal_timeseries),
        (global_co2_regression_wf.output.regressor_beta, ProcessWorkflow.output.global_co2_beta),
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
        (correlate_global_regressor_timeseries.output.timeshift_maxcorr, ProcessWorkflow.output.global_regressor_timeshift),
        # correlation bounds
        (ProcessWorkflow.input.align_regressor_lower_bound, add_none_lower.input.x),
        (ProcessWorkflow.input.align_regressor_upper_bound, add_none_upper.input.x),
        (correlate_global_regressor_timeseries.output.timeshift_maxcorr, add_none_lower.input.y),
        (correlate_global_regressor_timeseries.output.timeshift_maxcorr, add_none_upper.input.y),
        (add_none_lower.output.output, ProcessWorkflow.output.align_regressor_absolute_lower_bound),
        (add_none_upper.output.output, ProcessWorkflow.output.align_regressor_absolute_upper_bound),
    ),
    description="setup regression wf"
)


#%%
############################################################################################
# iterative cvr wf
############################################################################################

##############################################
# iterate correlate over bold timeseries
##############################################
iterate_correlate_wf = IteratingNode(Correlate(), iterating_inputs="timeseries_a", iterating_name="bold", description="iterate correlate bold timeseries").setDefaultInputs(boldIter_nr_parallel_processes = -1)

##############################################
# filter timeshifts
##############################################
filter_timeshifts = FilterTimeshifts(description="filter timeshifts")

##############################################
# iterate align downsample over bold timeseries
##############################################
iterate_align_downsample_wf = IteratingNode(align_downsample_wf.copy(), iterating_inputs=("ref_timeseries", "timeshift"), iterating_name="bold", description="iterate align, downsample bold timeseries").setDefaultInputs(boldIter_nr_parallel_processes = -1)

##############################################
# iterate calculate cvr over bold timeseries
##############################################
iterate_regress = IteratingNode(RegressCVR(), iterating_inputs=("bold_ts", "regressor_timeseries"), iterating_name="bold", exclude_outputs=("design_matrix", "betas"), description="iterative calculate cvr").setDefaultInputs(boldIter_nr_parallel_processes = -1)

# %%
iterate_cvr_wf = ProcessWorkflow(
    (
        # iterate correlate
        (ProcessWorkflow.input.nr_parallel_processes, iterate_correlate_wf.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_correlate_wf.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_correlate_wf.input.boldIter_timeseries_a),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_correlate_wf.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, iterate_correlate_wf.input.time_step),
        (ProcessWorkflow.input.align_regressor_absolute_lower_bound, iterate_correlate_wf.input.lower_limit),
        (ProcessWorkflow.input.align_regressor_absolute_upper_bound, iterate_correlate_wf.input.upper_limit),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_wf.input.bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_wf.input.window),
        (ProcessWorkflow.input.correlation_phat, iterate_correlate_wf.input.phat),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_correlate_wf.input.multi_peak_strategy),
        (ProcessWorkflow.input.global_regressor_timeshift, iterate_correlate_wf.input.ref_timeshift),
        (iterate_correlate_wf.output.all - iterate_correlate_wf.output.boldIter_timeshift_maxcorr, ProcessWorkflow.output._),
        # filter timeshift
        (ProcessWorkflow.input.timeseries_masker, filter_timeshifts.input.timeseries_masker),
        (ProcessWorkflow.input.filter_timeshifts_maxcorr_threshold, filter_timeshifts.input.maxcorr_threshold),
        (ProcessWorkflow.input.filter_timeshifts_size, filter_timeshifts.input.size),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, filter_timeshifts.input.filter_type),
        (iterate_correlate_wf.output.boldIter_timeshift_maxcorr, filter_timeshifts.input.timeshifts),
        (iterate_correlate_wf.output.boldIter_maxcorr, filter_timeshifts.input.maxcorrs),
        (filter_timeshifts.output.filtered_timeshifts, ProcessWorkflow.output.boldIter_timeshift_maxcorr),
        # iterate align downsample 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_align_downsample_wf.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_align_downsample_wf.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_align_downsample_wf.input.boldIter_ref_timeseries),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_align_downsample_wf.input.align_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_align_downsample_wf.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_align_downsample_wf.input.down_sampling_factor),
        (filter_timeshifts.output.filtered_timeshifts, iterate_align_downsample_wf.input.boldIter_timeshift),
        (iterate_align_downsample_wf.output.all / iterate_align_downsample_wf.output["boldIter_down_sampled_ref_timeseries", "boldIter_down_sampled_aligned_timeseries"], ProcessWorkflow.output._),
        (iterate_align_downsample_wf.output.boldIter_down_sampled_ref_timeseries, ProcessWorkflow.output.boldIter_down_sampled_bold_signal_ts),
        (iterate_align_downsample_wf.output.boldIter_down_sampled_aligned_timeseries, ProcessWorkflow.output.boldIter_down_sampled_aligned_regressor_signal_timeseries),
        # iterate calculate cvr
        (ProcessWorkflow.input.nr_parallel_processes, iterate_regress.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_regress.input.boldIter_show_pbar),
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, iterate_regress.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, iterate_regress.input.confound_regressor_correlation_threshold),
        (iterate_align_downsample_wf.output.boldIter_down_sampled_ref_timeseries, iterate_regress.input.boldIter_bold_ts),
        (iterate_align_downsample_wf.output.boldIter_down_sampled_aligned_timeseries, iterate_regress.input.boldIter_regressor_timeseries),
        (iterate_regress.output.all - iterate_regress.output.boldIter_predictions, ProcessWorkflow.output._),
        (iterate_regress.output.boldIter_predictions, ProcessWorkflow.output.boldIter_down_sampled_bold_signal_predictions)
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
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("nr_parallel_processes", "show_pbar", "sample_time", "down_sampling_factor", "maxcorr_bipolar", "correlation_window", "correlation_phat", "correlation_multi_peak_strategy", "timeseries_masker", "filter_timeshifts_maxcorr_threshold", "filter_timeshifts_size", "filter_timeshifts_filter_type", "confound_regressor_correlation_threshold")]),
        (setup_regression_wf.output.bold_signal_timeseries, iterate_cvr_wf.input.bold_signal_timeseries),
        (setup_regression_wf.output.regressor_signal_timeseries, iterate_cvr_wf.input.regressor_signal_timeseries),
        (setup_regression_wf.output.down_sampled_regression_confounds_signal_df, iterate_cvr_wf.input.down_sampled_regression_confounds_signal_df),
        (setup_regression_wf.output.align_regressor_absolute_lower_bound, iterate_cvr_wf.input.align_regressor_absolute_lower_bound),
        (setup_regression_wf.output.align_regressor_absolute_upper_bound, iterate_cvr_wf.input.align_regressor_absolute_upper_bound),
        (setup_regression_wf.output.global_regressor_timeshift, iterate_cvr_wf.input.global_regressor_timeshift),
        (iterate_cvr_wf.output.all, ProcessWorkflow.output._),
        # compute down sampled sample time
        (ProcessWorkflow.input.sample_time * ProcessWorkflow.input.down_sampling_factor, ProcessWorkflow.output.down_sampled_sample_time)
    ),
    description="regression wf"
)
# %%
