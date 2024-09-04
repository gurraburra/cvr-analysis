# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.modalities_study.workflows.utils.signal_processing import DownsampleTimeSeries
from cvr_analysis.modalities_study.workflows.utils.data_computation import Correlate, AlignTimeSeries, RegressCVR, PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries, RMSTimeSeries
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
            (ProcessWorkflow.input.sample_time, global_autocorrelation.input.time_step),
            (ProcessWorkflow.input.global_signal_timeseries, global_autocorrelation.input[("timeseries_a", "timeseries_b")]),
            (ValueNode(0).output.value, global_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, global_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, global_autocorrelation.input.bipolar),
            (global_autocorrelation.output.timeshifts, ProcessWorkflow.output.global_signal_autocorrelation_timeshifts),
            (global_autocorrelation.output.correlations, ProcessWorkflow.output.global_signal_autocorrelation_correlations),
            # global signal rms
            (ProcessWorkflow.input.co2_signal_timeseries, co2_rms.input.timeseries),
            (co2_rms.output.rms, ProcessWorkflow.output.co2_signal_rms),
            # co2 autocorrelation
            (ProcessWorkflow.input.correlation_window, co2_autocorrelation.input.window),
            (ProcessWorkflow.input.sample_time, co2_autocorrelation.input.time_step),
            (ProcessWorkflow.input.co2_signal_timeseries, co2_autocorrelation.input[("timeseries_a", "timeseries_b")]),
            (ValueNode(0).output.value, co2_autocorrelation.input.lower_limit),
            (ValueNode(None).output.value, co2_autocorrelation.input.upper_limit),
            (ValueNode(True).output.value, co2_autocorrelation.input.bipolar),
            (co2_autocorrelation.output.timeshifts, ProcessWorkflow.output.co2_signal_autocorrelation_timeshifts),
            (co2_autocorrelation.output.correlations, ProcessWorkflow.output.co2_signal_autocorrelation_correlations),
        )
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
# correlate, align, downsample wf
##--##--##--##--##--##--##--##--##--##--##--##
# correlate bold and regressor
correlate_bold_regressor_timeseries = Correlate(description="correlate single bold timeseries")

# align regressor
align_regressor_timeseries = AlignTimeSeries(description="align regressor timeseries")

# downsample
down_sample_bold_timeseries = DownsampleTimeSeries(description="down sample single bold timeseries")
down_sample_regressor_timeseries = DownsampleTimeSeries(description="down sample regressor timeseries")

correlate_align_downsample_wf = ProcessWorkflow(
    (
        # correlate bold regressor (using bold_ts instead bold_timeseries to signal single timeseries)
        (ProcessWorkflow.input.bold_signal_ts, correlate_bold_regressor_timeseries.input.timeseries_a),
        (ProcessWorkflow.input.regressor_signal_timeseries, correlate_bold_regressor_timeseries.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, correlate_bold_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.align_regressor_absolute_lower_bound, correlate_bold_regressor_timeseries.input.lower_limit),
        (ProcessWorkflow.input.align_regressor_absolute_upper_bound, correlate_bold_regressor_timeseries.input.upper_limit),
        (ProcessWorkflow.input.maxcorr_bipolar, correlate_bold_regressor_timeseries.input.bipolar),
        (ProcessWorkflow.input.correlation_window, correlate_bold_regressor_timeseries.input.window),
        (correlate_bold_regressor_timeseries.output.all, ProcessWorkflow.output._),
        # align regressor
        (ProcessWorkflow.input.regressor_signal_timeseries, align_regressor_timeseries.input.timeseries),
        (ProcessWorkflow.input.sample_time, align_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.bold_signal_ts.shape[0], align_regressor_timeseries.input.length),
        (ValueNode(True).output.value, align_regressor_timeseries.input.fill_nan),
        (correlate_bold_regressor_timeseries.output.timeshift_maxcorr, align_regressor_timeseries.input.timeshift),
        # down sample bold_ts
        (ProcessWorkflow.input.down_sampling_factor, down_sample_bold_timeseries.input.down_sampling_factor),
        (ProcessWorkflow.input.bold_signal_ts, down_sample_bold_timeseries.input.timeseries),
        (down_sample_bold_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_bold_signal_ts),
        # down sample regressor series
        (ProcessWorkflow.input.down_sampling_factor, down_sample_regressor_timeseries.input.down_sampling_factor),
        (align_regressor_timeseries.output.aligned_timeseries, down_sample_regressor_timeseries.input.timeseries),
        (down_sample_regressor_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_aligned_regressor_signal_timeseries)
    ),
    description="correlate, align, down sample workflow"
)

# %%
##--##--##--##--##--##--##--##--##--##--##--##
# calculate cvr
##--##--##--##--##--##--##--##--##--##--##--##
# regress bold and regressor
bold_regression = RegressCVR(description="regress single bold timeseries")
global_co2_correlate_align_downsample = correlate_align_downsample_wf.copy()

global_co2_regression_wf = ProcessWorkflow(
    (
        # correlate align downsample
        (ProcessWorkflow.input._, global_co2_correlate_align_downsample.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.co2_signal_timeseries, global_co2_correlate_align_downsample.input.regressor_signal_timeseries),
        (ProcessWorkflow.input.global_signal_timeseries, global_co2_correlate_align_downsample.input.bold_signal_ts),
        (ProcessWorkflow.input.correlation_window, global_co2_correlate_align_downsample.input.correlation_window),
        (ValueNode(False).output.value, global_co2_correlate_align_downsample.input.maxcorr_bipolar),
        (ValueNode(None).output.value, global_co2_correlate_align_downsample.input.align_regressor_absolute_lower_bound),
        (ValueNode(None).output.value, global_co2_correlate_align_downsample.input.align_regressor_absolute_upper_bound),
        (global_co2_correlate_align_downsample.output.all, ProcessWorkflow.output._),
        # regress
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, bold_regression.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, bold_regression.input.confound_regressor_correlation_threshold),
        (global_co2_correlate_align_downsample.output.down_sampled_bold_signal_ts, bold_regression.input.bold_ts),
        (global_co2_correlate_align_downsample.output.down_sampled_aligned_regressor_signal_timeseries, bold_regression.input.regressor_timeseries),
        (bold_regression.output.all / bold_regression.output["design_matrix", "betas", "predictions"], ProcessWorkflow.output._),
    ),
    description="global regressor timeshift and beta wf"
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
        (ProcessWorkflow.input._, global_co2_regression_wf.input[("sample_time", "down_sampling_factor", "correlation_window")]),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, global_co2_regression_wf.input.confound_regressor_correlation_threshold),
        (signal_timeseries_wf.output.co2_signal_timeseries, global_co2_regression_wf.input.co2_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, global_co2_regression_wf.input.global_signal_timeseries),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_signal_df, global_co2_regression_wf.input.down_sampled_regression_confounds_signal_df),
        (global_co2_regression_wf.output.timeshift_maxcorr, ProcessWorkflow.output.global_co2_timeshift_maxcorr),
        (global_co2_regression_wf.output.maxcorr, ProcessWorkflow.output.global_co2_maxcorr),
        (global_co2_regression_wf.output.timeshifts, ProcessWorkflow.output.global_co2_timeshifts),
        (global_co2_regression_wf.output.correlations, ProcessWorkflow.output.global_co2_correlations),
        (global_co2_regression_wf.output.down_sampled_bold_signal_ts, ProcessWorkflow.output.down_sampled_global_signal_timeseries),
        (global_co2_regression_wf.output.down_sampled_aligned_regressor_signal_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_co2_signal_timeseries),
        (global_co2_regression_wf.output.regressor_beta, ProcessWorkflow.output.global_co2_beta),
        # initial global regressor alignment
        (ProcessWorkflow.input.sample_time, correlate_global_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.initial_global_align_lower_bound, correlate_global_regressor_timeseries.input.lower_limit),
        (ProcessWorkflow.input.initial_global_align_upper_bound, correlate_global_regressor_timeseries.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, correlate_global_regressor_timeseries.input.window),
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
# iterate correlate align over bold timeseries
##############################################

iterate_correlate_align_downsample_wf = IteratingNode(correlate_align_downsample_wf.copy(), iterating_inputs="bold_signal_ts", iterating_name="bold", description="iterate correlate, align, downsample bold timeseries")

##############################################
# iterate calculate cvr over bold timeseries
##############################################

iterate_regress = IteratingNode(bold_regression.copy(), iterating_inputs=("bold_ts", "regressor_timeseries"), iterating_name="bold", exclude_outputs=("design_matrix", "betas"), description="iterative calculate cvr")

# %%
iterate_cvr_wf = ProcessWorkflow(
    (
        # iterate align correlate align downsample 
        (ProcessWorkflow.input.nr_parallel_processes, iterate_correlate_align_downsample_wf.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_correlate_align_downsample_wf.input.boldIter_show_pbar),
        (ProcessWorkflow.input.bold_signal_timeseries.T, iterate_correlate_align_downsample_wf.input.boldIter_bold_signal_ts),
        (ProcessWorkflow.input.regressor_signal_timeseries, iterate_correlate_align_downsample_wf.input.regressor_signal_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_correlate_align_downsample_wf.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_correlate_align_downsample_wf.input.down_sampling_factor),
        (ProcessWorkflow.input.align_regressor_absolute_lower_bound, iterate_correlate_align_downsample_wf.input.align_regressor_absolute_lower_bound),
        (ProcessWorkflow.input.align_regressor_absolute_upper_bound, iterate_correlate_align_downsample_wf.input.align_regressor_absolute_upper_bound),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_align_downsample_wf.input.maxcorr_bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_align_downsample_wf.input.correlation_window),
        (iterate_correlate_align_downsample_wf.output.all, ProcessWorkflow.output._),
        # iterate calculate cvr
        (ProcessWorkflow.input.nr_parallel_processes, iterate_regress.input.boldIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_regress.input.boldIter_show_pbar),
        (ProcessWorkflow.input.down_sampled_regression_confounds_signal_df, iterate_regress.input.confounds_df),
        (ProcessWorkflow.input.confound_regressor_correlation_threshold, iterate_regress.input.confound_regressor_correlation_threshold),
        (iterate_correlate_align_downsample_wf.output.boldIter_down_sampled_bold_signal_ts, iterate_regress.input.boldIter_bold_ts),
        (iterate_correlate_align_downsample_wf.output.boldIter_down_sampled_aligned_regressor_signal_timeseries, iterate_regress.input.boldIter_regressor_timeseries),
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
        (ProcessWorkflow.input._, iterate_cvr_wf.input[("nr_parallel_processes", "show_pbar", "sample_time", "down_sampling_factor", "maxcorr_bipolar", "correlation_window", "confound_regressor_correlation_threshold")]),
        (setup_regression_wf.output.bold_signal_timeseries, iterate_cvr_wf.input.bold_signal_timeseries),
        (setup_regression_wf.output.regressor_signal_timeseries, iterate_cvr_wf.input.regressor_signal_timeseries),
        (setup_regression_wf.output.down_sampled_regression_confounds_signal_df, iterate_cvr_wf.input.down_sampled_regression_confounds_signal_df),
        (setup_regression_wf.output.align_regressor_absolute_lower_bound, iterate_cvr_wf.input.align_regressor_absolute_lower_bound),
        (setup_regression_wf.output.align_regressor_absolute_upper_bound, iterate_cvr_wf.input.align_regressor_absolute_upper_bound),
        (iterate_cvr_wf.output.all, ProcessWorkflow.output._),
        # compute down sampled sample time
        (ProcessWorkflow.input.sample_time * ProcessWorkflow.input.down_sampling_factor, ProcessWorkflow.output.down_sampled_sample_time)
    ),
    description="regression wf"
)
# %%
