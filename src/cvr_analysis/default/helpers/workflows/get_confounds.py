# %%
# process control
from process_control import *
from cvr_analysis.default.helpers.classes.signal_processing import DownsampleTimeSeries
from cvr_analysis.default.helpers.classes.confounds import MotionConfounds, SpikeConfounds, DriftConfounds

#%%
# %%
##############################################
# get regression confounds wf
##############################################
# downsample confounds
down_sample_confounds_df = DownsampleTimeSeries(description="down sample confounds df")
down_sample_depvars_timeseries = DownsampleTimeSeries(description="down sample depvars ts")
down_sample_global_timeseries = DownsampleTimeSeries(description="down sample global ts")

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
        # down sample confounds
        (ProcessWorkflow.input.down_sampling_factor, down_sample_confounds_df.input.down_sampling_factor),
        (ProcessWorkflow.input.confounds_df, down_sample_confounds_df.input.timeseries),
        # down sample depvars ts (need to downsample first so spikes don't get lost)
        (ProcessWorkflow.input.down_sampling_factor, down_sample_depvars_timeseries.input.down_sampling_factor),
        (ProcessWorkflow.input.depvars_timeseries, down_sample_depvars_timeseries.input.timeseries),
        # down sample global ts (need to downsample first so spikes don't get lost)
        (ProcessWorkflow.input.down_sampling_factor, down_sample_global_timeseries.input.down_sampling_factor),
        (ProcessWorkflow.input.global_timeseries, down_sample_global_timeseries.input.timeseries),
        # motion confounds
        (ProcessWorkflow.input.include_motion_confounds, cond_load_motion_confounds.input.include_motion_confounds),
        (ProcessWorkflow.input.motion_derivatives, cond_load_motion_confounds.input.derivatives),
        (ProcessWorkflow.input.motion_powers, cond_load_motion_confounds.input.powers),
        (ValueNode(None).output.value, cond_load_motion_confounds.input.old_confounds_df),
        (down_sample_confounds_df.output.down_sampled_timeseries, cond_load_motion_confounds.input.confounds_df),
        # drift confounds
        (ProcessWorkflow.input._, cond_load_drift_confounds.input[("include_drift_confounds", "drift_high_pass", "drift_model", "drift_order")]),
        (ProcessWorkflow.input.time_step, cond_load_drift_confounds.input.tr),
        (ValueNode(0.5).output.value, cond_load_drift_confounds.input.ref_slice),
        (down_sample_depvars_timeseries.output.down_sampled_timeseries.shape[0], cond_load_drift_confounds.input.nr_measurements),
        (cond_load_motion_confounds.output.confounds_df, cond_load_drift_confounds.input.old_confounds_df),
        # spike confounds
        (down_sample_depvars_timeseries.output.down_sampled_timeseries, cond_load_spike_confounds.input.depvars_timeseries),
        (down_sample_global_timeseries.output.down_sampled_timeseries, cond_load_spike_confounds.input.global_timeseries),
        (ProcessWorkflow.input.include_spike_confounds, cond_load_spike_confounds.input.include_spike_confounds),
        (ProcessWorkflow.input.spike_diff_cutoff, cond_load_spike_confounds.input.difference_cutoff),
        (ProcessWorkflow.input.spike_global_cutoff, cond_load_spike_confounds.input.global_cutoff),
        (cond_load_drift_confounds.output.confounds_df, cond_load_spike_confounds.input.old_confounds_df),
        # add constant confound
        (cond_load_spike_confounds.output.confounds_df, add_constant_confound.input.confounds_df),
        (add_constant_confound.output.confounds_df, ProcessWorkflow.output.down_sampled_regression_confounds_df)
    ), description="get regression confounds wf"
)