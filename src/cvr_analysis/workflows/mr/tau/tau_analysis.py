# %%
import numpy as np

# process control
from process_control import *
from cvr_analysis.workflows.utils.data_computation import PercentageChangeTimeSeries, StandardizeTimeSeries, BaselineTimeSeries
from cvr_analysis.workflows.utils.dev.confounds import MotionConfounds, DriftConfounds, SpikeConfounds
from cvr_analysis.workflows.mr.tau.tau_delay import TauDelay

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

# co2
co2_baseline = BaselineTimeSeries(description="baseline co2 timeseries")
co2_signal_wf = ProcessWorkflow(
    (
        (ProcessWorkflow.input.co2_timeseries, co2_baseline.input.timeseries),
        (ProcessWorkflow.input.baseline_strategy, co2_baseline.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, co2_baseline.input.time_step),
        (ProcessWorkflow.input.co2_timeseries - co2_baseline.output.baseline, ProcessWorkflow.output.co2_signal_timeseries),
    ),
    description="create co2 signal"
)
cond_co2_signal = ConditionalNode("use_co2_regressor", {True : co2_signal_wf, False : None}, description="conditionally compute co2 signal")

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
        (ProcessWorkflow.input.use_co2_regressor, cond_co2_signal.input.use_co2_regressor),
        (ProcessWorkflow.input.co2_timeseries, cond_co2_signal.input.co2_timeseries),
        (ProcessWorkflow.input.baseline_strategy, cond_co2_signal.input.baseline_strategy),
        (ProcessWorkflow.input.sample_time, cond_co2_signal.input.sample_time),
        (cond_co2_signal.output.co2_signal_timeseries, ProcessWorkflow.output.co2_signal_timeseries),
        # confounds
        (ProcessWorkflow.input.confounds_df, confounds_std.input.timeseries),
        (confounds_std.output.standardized_timeseries, ProcessWorkflow.output.confounds_signal_df),
    ),
    description="convert timeseries into signals"
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

# %%
##############################################
# get regression confounds wf
##############################################

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

get_tau_confounds_wf = ProcessWorkflow(
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
        (add_constant_confound.output.confounds_df, ProcessWorkflow.output.confounds_df)
    ), description="get regression confounds wf"
)

# %% setup tau analysis
setup_tau_analysis_wf = ProcessWorkflow(
    (
        # timeseries signal wf
        (ProcessWorkflow.input._, signal_timeseries_wf.input.all),
        (signal_timeseries_wf.output.all, ProcessWorkflow.output._),
        # choose regressor
        (ProcessWorkflow.input._, choose_regressor.input.use_co2_regressor),
        (signal_timeseries_wf.output.co2_signal_timeseries, choose_regressor.input.co2_signal_timeseries),
        (signal_timeseries_wf.output.global_signal_timeseries, choose_regressor.input.global_signal_timeseries),
        (choose_regressor.output.signal_timeseries, ProcessWorkflow.output.regressor_signal_timeseries),
        # get tau confounds wf
        (ProcessWorkflow.input._, get_tau_confounds_wf.input[("drift_high_pass", "drift_model", "drift_order", "include_drift_confounds", "include_motion_confounds", "include_spike_confounds", "motion_derivatives", "motion_powers", "spike_diff_cutoff", "spike_global_cutoff")]),
        (ProcessWorkflow.input.sample_time, get_tau_confounds_wf.input.time_step),
        (signal_timeseries_wf.output.confounds_signal_df, get_tau_confounds_wf.input.confounds_signal_df),
        (signal_timeseries_wf.output.bold_signal_timeseries, get_tau_confounds_wf.input.bold_signal_timeseries),
        (get_tau_confounds_wf.output.confounds_df, ProcessWorkflow.output.tau_confounds_df),
    ),
    description="setup tau analysis wf"
)


#%%
############################################################################################
# tau_delay
############################################################################################
tau_delay = TauDelay(description="tau analysis")
# %%

tau_analysis_wf = ProcessWorkflow(
    (
        # regression setup
        (ProcessWorkflow.input._, setup_tau_analysis_wf.input.all),
        (setup_tau_analysis_wf.output.all, ProcessWorkflow.output._),
        # tau_delay
        (ProcessWorkflow.input.delay_lower_limit, tau_delay.input.lag_min),
        (ProcessWorkflow.input.delay_upper_limit, tau_delay.input.lag_max),
        (ProcessWorkflow.input.sample_time, tau_delay.input.time_step),
        (ProcessWorkflow.input._, tau_delay.input[("nr_tau", "tau_min", "tau_max", "tau_log_space", "phat")]),
        (setup_tau_analysis_wf.output.tau_confounds_df, tau_delay.input.confounds),
        (setup_tau_analysis_wf.output.bold_signal_timeseries, tau_delay.input.signals),
        (setup_tau_analysis_wf.output.regressor_signal_timeseries, tau_delay.input.probe),
        (tau_delay.output.beta, ProcessWorkflow.output.cvr),
        (tau_delay.output.delay, ProcessWorkflow.output.delay),
        (tau_delay.output.tau, ProcessWorkflow.output.tau),
        (tau_delay.output.r2, ProcessWorkflow.output.r2),
        (tau_delay.output.sse, ProcessWorkflow.output.sse),
    ),
    description="tau analysis wf"
)
# %%
