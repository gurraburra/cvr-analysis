# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.modalities_study.workflows.utils.signal_processing import DownsampleTimeSeries
from cvr_analysis.modalities_study.workflows.utils.data_computation import NormTimeSeries, BaselinePlateau, Correlate, AlignTimeSeries, RegressCVR, CVRAmplitude
from cvr_analysis.modalities_study.workflows.utils.confounds import MotionConfounds

# %%

##########################################################################################################################################
# regression wf
##########################################################################################################################################

############################################################################################
# regression setup
############################################################################################

##############################################
# prepare and compute regressor characterstics
##############################################
norm_global_timeseries = NormTimeSeries(description="norm global timeseries")
compute_baseline_plateau_co2_timeseries = BaselinePlateau(description="compute baseline and plateau for co2 timeseries")
prepare_regressors = ProcessWorkflow(
    (
        # global regressor
        (ProcessWorkflow.input.bold_timeseries.mean(axis = 1), norm_global_timeseries.input.timeseries),
        (norm_global_timeseries.output.normed_timeseries, ProcessWorkflow.output.normed_global_timeseries),
        (norm_global_timeseries.output.normed_baseline, ProcessWorkflow.output.normed_global_baseline),
        (norm_global_timeseries.output.normed_plateau, ProcessWorkflow.output.normed_global_plateau),
        (ProcessWorkflow.input.bold_timeseries.mean(axis = 1).std(), ProcessWorkflow.output.normed_global_std),
        # co2 regressor
        (ProcessWorkflow.input.co2_timeseries, compute_baseline_plateau_co2_timeseries.input.timeseries),
        (ProcessWorkflow.input.co2_timeseries, ProcessWorkflow.output.co2_timeseries),
        (compute_baseline_plateau_co2_timeseries.output.baseline, ProcessWorkflow.output.co2_baseline),
        (compute_baseline_plateau_co2_timeseries.output.plateau, ProcessWorkflow.output.co2_plateau),
        (ProcessWorkflow.input.co2_timeseries.std(), ProcessWorkflow.output.co2_std),
    ),
    description="prepare potential regressor and compute characterstics"
)
# %%

##############################################
# choose regressor
##############################################
regressor_dummy_wf = ProcessWorkflow(
    (
        (ProcessWorkflow.input.timeseries, ProcessWorkflow.output.timeseries),
        (ProcessWorkflow.input.baseline, ProcessWorkflow.output.baseline),
        (ProcessWorkflow.input.plateau, ProcessWorkflow.output.plateau),
    )
)
co2_dummy = regressor_dummy_wf.copy()
global_dummy = regressor_dummy_wf.copy()
dummy_input_mapping = {
                    "co2_timeseries"     : co2_dummy.input.timeseries, 
                    "co2_baseline"       : co2_dummy.input.baseline, 
                    "co2_plateau"        : co2_dummy.input.plateau, 
                    "global_timeseries"  : global_dummy.input.timeseries, 
                    "global_baseline"    : global_dummy.input.baseline, 
                    "global_plateau"     : global_dummy.input.plateau, 
                }
choose_regressor = ConditionalNode("use_co2_regressor", {True : co2_dummy, False : global_dummy}, default_condition=True, input_mapping = dummy_input_mapping, description="choose regressor")

get_regressor_wf = ProcessWorkflow(
    (
        # global timeseries: norm and compute baseline, plateau and std
        (ProcessWorkflow.input.bold_timeseries.mean(axis = 1), norm_global_timeseries.input.timeseries),
        (norm_global_timeseries.output.normed_timeseries, ProcessWorkflow.output.normed_global_timeseries),
        (norm_global_timeseries.output.normed_baseline, ProcessWorkflow.output.normed_global_baseline),
        (norm_global_timeseries.output.normed_plateau, ProcessWorkflow.output.normed_global_plateau),
        (norm_global_timeseries.output.normed_timeseries.std(), ProcessWorkflow.output.normed_global_std),
        # global timeseries: norm and compute baseline, plateau and std
        (ProcessWorkflow.input.co2_timeseries, compute_baseline_plateau_co2_timeseries.input.timeseries),
        (compute_baseline_plateau_co2_timeseries.output.baseline, ProcessWorkflow.output.co2_baseline),
        (compute_baseline_plateau_co2_timeseries.output.plateau, ProcessWorkflow.output.co2_plateau),
        (ProcessWorkflow.input.co2_timeseries.std(), ProcessWorkflow.output.co2_std),
        # chhose regressor
        (ProcessWorkflow.input.use_co2_regressor, choose_regressor.input.use_co2_regressor),
        (ProcessWorkflow.input.co2_timeseries, choose_regressor.input.co2_timeseries),
        (compute_baseline_plateau_co2_timeseries.output.baseline, choose_regressor.input.co2_baseline),
        (compute_baseline_plateau_co2_timeseries.output.plateau, choose_regressor.input.co2_plateau),
        (norm_global_timeseries.output.normed_timeseries, choose_regressor.input.global_timeseries),
        (norm_global_timeseries.output.normed_baseline, choose_regressor.input.global_baseline),
        (norm_global_timeseries.output.normed_plateau, choose_regressor.input.global_plateau),
        (choose_regressor.output.timeseries, ProcessWorkflow.output.regressor_timeseries),
        (choose_regressor.output.baseline, ProcessWorkflow.output.regressor_baseline),
        (choose_regressor.output.plateau, ProcessWorkflow.output.regressor_plateau),
    ),
    "Get regressor workflow"
)

# %%
##############################################
# get regression confounds wf
##############################################

# load motion confounds
load_motion_confounds = MotionConfounds(description="get motion confounds")

# threshold motion regressor maxcorr
motion_regressor_maxcorr = IteratingNode(Correlate(description="correlate regressor with motion confound"), ("timeseries_a", ), "motion", exclude_outputs=("timeshift_maxcorr", "timeshifts", "correlations"), show_pbar=False, description="compute maxcorr between regresssor and motion confounds")
get_confounds_below_thr = CustomNode(lambda motion_confounds_df, motion_regressor_maxcorr, motion_regressor_correlation_threshold = None : 
                                        motion_confounds_df.loc[:, np.abs(motion_regressor_maxcorr) < motion_regressor_correlation_threshold] \
                                            if motion_regressor_correlation_threshold is not None else motion_confounds_df, 
                                        outputs="thresholded_motion_confounds_df", description="threshold motion confounds"
                                    )

# downsample confounds
down_sample_confounds_df = DownsampleTimeSeries(description="down sample confounds df")

get_regression_confounds_wf = ProcessWorkflow(
    (
        # motion confounds
        (ProcessWorkflow.input.confounds_df, load_motion_confounds.input.confounds_df),
        (ProcessWorkflow.input.motion_derivatives, load_motion_confounds.input.derivatives),
        (ProcessWorkflow.input.motion_powers, load_motion_confounds.input.powers),
        (ValueNode(True).output.value, load_motion_confounds.input.standardize),
        (load_motion_confounds.output.motion_confounds_df.columns, ProcessWorkflow.output.motion_confound_names),
        # motion regressor maxcorr
        (ProcessWorkflow.input.regressor_timeseries, motion_regressor_maxcorr.input.timeseries_b),
        (ProcessWorkflow.input.correlation_window, motion_regressor_maxcorr.input.window),
        (load_motion_confounds.output.motion_confounds_df.to_numpy().T, motion_regressor_maxcorr.input.motionIter_timeseries_a),
        (ValueNode(1).output.value, motion_regressor_maxcorr.input.time_step),
        (ValueNode(None).output.value, motion_regressor_maxcorr.input.lower_limit),
        (ValueNode(None).output.value, motion_regressor_maxcorr.input.upper_limit),
        (ValueNode(True).output.value, motion_regressor_maxcorr.input.bipolar),
        (motion_regressor_maxcorr.output.motionIter_maxcorr, ProcessWorkflow.output.motion_regressor_maxcorr),
        # get confounds below threshold
        (ProcessWorkflow.input.motion_regressor_correlation_threshold, get_confounds_below_thr.input.motion_regressor_correlation_threshold),
        (load_motion_confounds.output.motion_confounds_df, get_confounds_below_thr.input.motion_confounds_df),
        (motion_regressor_maxcorr.output.motionIter_maxcorr, get_confounds_below_thr.input.motion_regressor_maxcorr),
        # down sample confounds
        (ProcessWorkflow.input.down_sampling_factor, down_sample_confounds_df.input.down_sampling_factor),
        (get_confounds_below_thr.output.thresholded_motion_confounds_df, down_sample_confounds_df.input.timeseries),
        (down_sample_confounds_df.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_regression_confounds_df)
        
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
        (ProcessWorkflow.input.bold_ts, correlate_bold_regressor_timeseries.input.timeseries_a),
        (ProcessWorkflow.input.regressor_timeseries, correlate_bold_regressor_timeseries.input.timeseries_b),
        (ProcessWorkflow.input.sample_time, correlate_bold_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.align_regressor_lower_bound, correlate_bold_regressor_timeseries.input.lower_limit),
        (ProcessWorkflow.input.align_regressor_upper_bound, correlate_bold_regressor_timeseries.input.upper_limit),
        (ProcessWorkflow.input.maxcorr_bipolar, correlate_bold_regressor_timeseries.input.bipolar),
        (ProcessWorkflow.input.correlation_window, correlate_bold_regressor_timeseries.input.window),
        (correlate_bold_regressor_timeseries.output.all, ProcessWorkflow.output._),
        # align regressor
        (ProcessWorkflow.input.regressor_timeseries, align_regressor_timeseries.input.timeseries),
        (ProcessWorkflow.input.sample_time, align_regressor_timeseries.input.time_step),
        (ProcessWorkflow.input.bold_ts.shape[0], align_regressor_timeseries.input.length),
        (ValueNode(True).output.value, align_regressor_timeseries.input.fill_nan),
        (correlate_bold_regressor_timeseries.output.timeshift_maxcorr, align_regressor_timeseries.input.timeshift),
        # down sample bold_ts
        (ProcessWorkflow.input.down_sampling_factor, down_sample_bold_timeseries.input.down_sampling_factor),
        (ProcessWorkflow.input.bold_ts, down_sample_bold_timeseries.input.timeseries),
        (down_sample_bold_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_bold_ts),
        # down sample regressor series
        (ProcessWorkflow.input.down_sampling_factor, down_sample_regressor_timeseries.input.down_sampling_factor),
        (align_regressor_timeseries.output.aligned_timeseries, down_sample_regressor_timeseries.input.timeseries),
        (down_sample_regressor_timeseries.output.down_sampled_timeseries, ProcessWorkflow.output.down_sampled_aligned_regressor_timeseries)
    ),
    description="correlate, align, down sample workflow"
)

# %%
##--##--##--##--##--##--##--##--##--##--##--##
# calculate cvr
##--##--##--##--##--##--##--##--##--##--##--##
# regress bold and regressor
regress_bold_regressor_timeseries = RegressCVR(description="regress single bold timeseries")

# cvr amplitude
compute_cvr = CVRAmplitude(description="calculate cvr amplitude")

# calculate cvr
calculate_cvr_wf = ProcessWorkflow(
    (
        # regress
        (ProcessWorkflow.input.down_sampled_bold_ts, regress_bold_regressor_timeseries.input.bold_ts),
        (ProcessWorkflow.input.down_sampled_regressor_timeseries, regress_bold_regressor_timeseries.input.regressor_timeseries),
        (ProcessWorkflow.input.down_sampled_confounds_df, regress_bold_regressor_timeseries.input.confounds_df),
        (regress_bold_regressor_timeseries.output.all / regress_bold_regressor_timeseries.output["design_matrix", "betas", "predictions"], ProcessWorkflow.output._),
        (regress_bold_regressor_timeseries.output.predictions, ProcessWorkflow.output.down_sampled_bold_predictions),
        # cvr
        (ProcessWorkflow.input.regressor_baseline, compute_cvr.input.regressor_baseline),
        (regress_bold_regressor_timeseries.output.design_matrix, compute_cvr.input.design_matrix),
        (regress_bold_regressor_timeseries.output.betas, compute_cvr.input.betas),
        (compute_cvr.output.all, ProcessWorkflow.output._)
    ),
    description="calculate cvr workflow"
)

global_co2_regression_wf = ProcessWorkflow(
    (
        # correlate align downsample
        (ProcessWorkflow.input._, correlate_align_downsample_wf.input[("sample_time", "down_sampling_factor")]),
        (ProcessWorkflow.input.co2_timeseries, correlate_align_downsample_wf.input.regressor_timeseries),
        (ProcessWorkflow.input.global_timeseries, correlate_align_downsample_wf.input.bold_ts),
        (ProcessWorkflow.input.correlation_window, correlate_align_downsample_wf.input.correlation_window),
        (ValueNode(False).output.value, correlate_align_downsample_wf.input.maxcorr_bipolar),
        (ValueNode(None).output.value, correlate_align_downsample_wf.input.align_regressor_lower_bound),
        (ValueNode(None).output.value, correlate_align_downsample_wf.input.align_regressor_upper_bound),
        (correlate_align_downsample_wf.output.all, ProcessWorkflow.output._),
        # calculate cvr
        (ProcessWorkflow.input.down_sampled_regression_confounds_df, calculate_cvr_wf.input.down_sampled_confounds_df),
        (ProcessWorkflow.input.co2_baseline, calculate_cvr_wf.input.regressor_baseline),
        (correlate_align_downsample_wf.output.down_sampled_bold_ts, calculate_cvr_wf.input.down_sampled_bold_ts),
        (correlate_align_downsample_wf.output.down_sampled_aligned_regressor_timeseries, calculate_cvr_wf.input.down_sampled_regressor_timeseries),
        (calculate_cvr_wf.output.all, ProcessWorkflow.output._)
    ),
    description="global regressor timeshift and beta wf"
)

##############################################
# correlation bounds
##############################################
# timeshift global regressoor
no_timeshift = ValueNode(0,  description="timeshift reference for global regressor")
global_co2_timeshift = ProcessWorkflow(((ProcessWorkflow.input.global_co2_timeshift_maxcorr, ProcessWorkflow.output._),), description="global co2 timeshift dummy")

global_regressor_timeshift = ConditionalNode("use_co2_regressor", 
                                        {True : global_co2_timeshift, False : no_timeshift},
                                            output_mapping={"global_regressor_timeshift" : (no_timeshift.output.value, global_co2_timeshift.output.global_co2_timeshift_maxcorr)},
                                            description="global regressor timeshift")

# add if not None
add_none_lower = CustomNode(lambda x = None, y = None : x + y if x is not None and y is not None else None, description="add none")
add_none_upper = add_none_lower.copy()

# %%
setup_regression_wf = ProcessWorkflow(
    (
        # regressor characterstics
        (ProcessWorkflow.input._, prepare_regressors.input.all),
        (prepare_regressors.output.all, ProcessWorkflow.output._),
        # choose regressor
        (ProcessWorkflow.input._, choose_regressor.input.use_co2_regressor),
        (prepare_regressors.output.co2_timeseries, choose_regressor.input.co2_timeseries),
        (prepare_regressors.output.co2_baseline, choose_regressor.input.co2_baseline),
        (prepare_regressors.output.co2_plateau, choose_regressor.input.co2_plateau),
        (prepare_regressors.output.normed_global_timeseries, choose_regressor.input.global_timeseries),
        (prepare_regressors.output.normed_global_baseline, choose_regressor.input.global_baseline),
        (prepare_regressors.output.normed_global_plateau, choose_regressor.input.global_plateau),
        (choose_regressor.output.timeseries, ProcessWorkflow.output.regressor_timeseries),
        (choose_regressor.output.baseline, ProcessWorkflow.output.regressor_baseline),
        (choose_regressor.output.plateau, ProcessWorkflow.output.regressor_plateau),
        # confounds
        (ProcessWorkflow.input.confounds_df, get_regression_confounds_wf.input.confounds_df),
        (ProcessWorkflow.input.motion_regressor_correlation_threshold, get_regression_confounds_wf.input.motion_regressor_correlation_threshold),
        (ProcessWorkflow.input.down_sampling_factor, get_regression_confounds_wf.input.down_sampling_factor),
        (ProcessWorkflow.input.correlation_window, get_regression_confounds_wf.input.correlation_window),
        (choose_regressor.output.timeseries, get_regression_confounds_wf.input.regressor_timeseries),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_derivatives),
        (ValueNode(True).output.value, get_regression_confounds_wf.input.motion_powers),
        (get_regression_confounds_wf.output.all, ProcessWorkflow.output._),
        # global regressor beta
        (ProcessWorkflow.input._, global_co2_regression_wf.input[("sample_time", "down_sampling_factor", "correlation_window")]),
        (prepare_regressors.output.co2_timeseries, global_co2_regression_wf.input.co2_timeseries),
        (prepare_regressors.output.co2_baseline, global_co2_regression_wf.input.co2_baseline),
        (prepare_regressors.output.normed_global_timeseries, global_co2_regression_wf.input.global_timeseries),
        (get_regression_confounds_wf.output.down_sampled_regression_confounds_df, global_co2_regression_wf.input.down_sampled_regression_confounds_df),
        (global_co2_regression_wf.output.timeshift_maxcorr, ProcessWorkflow.output.global_co2_timeshift_maxcorr),
        (global_co2_regression_wf.output.maxcorr, ProcessWorkflow.output.global_co2_maxcorr),
        (global_co2_regression_wf.output.timeshifts, ProcessWorkflow.output.global_co2_timeshifts),
        (global_co2_regression_wf.output.correlations, ProcessWorkflow.output.global_co2_correlations),
        (global_co2_regression_wf.output.down_sampled_bold_ts, ProcessWorkflow.output.down_sampled_global_timeseries),
        (global_co2_regression_wf.output.down_sampled_aligned_regressor_timeseries, ProcessWorkflow.output.down_sampled_global_aligned_co2_timeseries),
        (global_co2_regression_wf.output.cvr_amplitude, ProcessWorkflow.output.global_co2_cvr_amplitude),
        (global_co2_regression_wf.output.regressor_beta, ProcessWorkflow.output.global_co2_beta),
        # timeshift reference
        (ProcessWorkflow.input._, global_regressor_timeshift.input.use_co2_regressor),
        (global_co2_regression_wf.output.timeshift_maxcorr, global_regressor_timeshift.input.global_co2_timeshift_maxcorr),
        (global_regressor_timeshift.output.global_regressor_timeshift, ProcessWorkflow.output.global_regressor_timeshift),
        # correlation bounds
        (ProcessWorkflow.input.align_regressor_lower_bound, add_none_lower.input.x),
        (ProcessWorkflow.input.align_regressor_upper_bound, add_none_upper.input.x),
        (global_regressor_timeshift.output.global_regressor_timeshift, add_none_lower.input.y),
        (global_regressor_timeshift.output.global_regressor_timeshift, add_none_upper.input.y),
        (add_none_lower.output.output, ProcessWorkflow.output.align_regressor_absolute_lower_bound),
        (add_none_upper.output.output, ProcessWorkflow.output.align_regressor_absolute_upper_bound),
    )
)


#%%
############################################################################################
# iterative regression analysis
############################################################################################

##############################################
# iterate correlate align over bold timeseries
##############################################

iterate_correlate_align_downsample_wf = IteratingNode(correlate_align_downsample_wf.copy(), iterating_inputs="bold_ts", iterating_name="bold", parallel_processing=True, show_pbar=True, description="iterate correlate, align, downsample bold timeseries")

##############################################
# iterate calculate cvr over bold timeseries
##############################################

iterate_calculate_cvr = IteratingNode(calculate_cvr_wf.copy(), iterating_inputs=("down_sampled_bold_ts", "down_sampled_regressor_timeseries"), iterating_name="bold", parallel_processing=True, show_pbar=True, description="iterative calculate cvr")

# %%
iterative_regression_wf = ProcessWorkflow(
    (
        # iterate align correlate align downsample 
        (ProcessWorkflow.input.bold_timeseries.T, iterate_correlate_align_downsample_wf.input.boldIter_bold_ts),
        (ProcessWorkflow.input.regressor_timeseries, iterate_correlate_align_downsample_wf.input.regressor_timeseries),
        (ProcessWorkflow.input.sample_time, iterate_correlate_align_downsample_wf.input.sample_time),
        (ProcessWorkflow.input.down_sampling_factor, iterate_correlate_align_downsample_wf.input.down_sampling_factor),
        (ProcessWorkflow.input.align_regressor_absolute_lower_bound, iterate_correlate_align_downsample_wf.input.align_regressor_lower_bound),
        (ProcessWorkflow.input.align_regressor_absolute_upper_bound, iterate_correlate_align_downsample_wf.input.align_regressor_upper_bound),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_align_downsample_wf.input.maxcorr_bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_align_downsample_wf.input.correlation_window),
        (iterate_correlate_align_downsample_wf.output.all, ProcessWorkflow.output._),
        # iterate calculate cvr
        (ProcessWorkflow.input.down_sampled_regression_confounds_df, iterate_calculate_cvr.input.down_sampled_confounds_df),
        (ProcessWorkflow.input.regressor_baseline, iterate_calculate_cvr.input.regressor_baseline),
        (iterate_correlate_align_downsample_wf.output.boldIter_down_sampled_bold_ts, iterate_calculate_cvr.input.boldIter_down_sampled_bold_ts),
        (iterate_correlate_align_downsample_wf.output.boldIter_down_sampled_aligned_regressor_timeseries, iterate_calculate_cvr.input.boldIter_down_sampled_regressor_timeseries),
        (iterate_calculate_cvr.output.all, ProcessWorkflow.output._)
    ),
    description="iterative over bold timeseries wf"
)

# %%

regression_wf = ProcessWorkflow(
    (
        # regression setup
        (ProcessWorkflow.input._, setup_regression_wf.input.all),
        (setup_regression_wf.output.all, ProcessWorkflow.output._),
        # iterative regression
        (ProcessWorkflow.input._, iterative_regression_wf.input[("bold_timeseries", "sample_time", "down_sampling_factor", "maxcorr_bipolar", "correlation_window")]),
        (setup_regression_wf.output.regressor_timeseries, iterative_regression_wf.input.regressor_timeseries),
        (setup_regression_wf.output.regressor_baseline, iterative_regression_wf.input.regressor_baseline),
        (setup_regression_wf.output.align_regressor_absolute_lower_bound, iterative_regression_wf.input.align_regressor_absolute_lower_bound),
        (setup_regression_wf.output.align_regressor_absolute_upper_bound, iterative_regression_wf.input.align_regressor_absolute_upper_bound),
        (setup_regression_wf.output.down_sampled_regression_confounds_df, iterative_regression_wf.input.down_sampled_regression_confounds_df),
        (iterative_regression_wf.output.all, ProcessWorkflow.output._),
        # compute down sampled sample time
        (ProcessWorkflow.input.sample_time / ProcessWorkflow.input.down_sampling_factor, ProcessWorkflow.output.down_sampled_sample_time)
    ),
    description="regression wf"
)
# %%
