from process_control import ProcessNode
import pandas as pd
import numpy as np
from nilearn import glm


class MotionConfounds(ProcessNode):
    outputs = ("confounds_df",)
    
    def _run(self, confounds_df : pd.DataFrame, derivatives : bool = True, powers : bool = True, old_confounds_df : pd.DataFrame = None) -> tuple:
        # return if false
        if confounds_df is None:
            return old_confounds_df, None
        # determine which confounds to include
        include = [""]
        if derivatives:
            include.append("_derivative1")
        if powers:
            include.append("_power2")
        if powers and derivatives:
            include.append("_derivative1_power2")

        # add all
        confound_names = []
        c_base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        for base in c_base:
            for add in include:
                confound_names.append(base + add)

        confound_selected = confounds_df.loc[:,confound_names]

        if old_confounds_df is not None:
            confounds = pd.concat((old_confounds_df.reset_index(drop=True), confound_selected.reset_index(drop=True)), axis = 1)
        else:
            confounds = confound_selected

        return confounds,
    
class DriftConfounds(ProcessNode):
    outputs = ("confounds_df", )
    
    def _run(self, tr : float, nr_measurements : int, ref_slice : float = 0.5, drift_model : str = 'polynomial', drift_order : int = 1, drift_high_pass : float = 0.01, old_confounds_df : pd.DataFrame = None) -> tuple:
        start_time = tr * ref_slice
        end_time = (nr_measurements - 1 + ref_slice) * tr
        frame_times = np.linspace(start_time, end_time, nr_measurements)

        drift_df = glm.first_level.make_first_level_design_matrix(frame_times, drift_model = drift_model, drift_order = drift_order, high_pass = drift_high_pass)

        if old_confounds_df is not None:
            drift_df = pd.concat((old_confounds_df.reset_index(drop=True), drift_df.reset_index(drop=True)), axis = 1)

        return drift_df, 

class SpikeConfounds(ProcessNode):
    outputs = ("confounds_df", )
    
    def _run(self, depvars_timeseries : np.ndarray, global_timeseries : np.ndarray, global_cutoff : float = 3, difference_cutoff : float = 3, old_confounds_df : pd.DataFrame = None) -> tuple:
        global_spikes = np.append(np.where(global_timeseries > global_timeseries.mean() + global_timeseries.std() * global_cutoff),
                                    np.where(global_timeseries < global_timeseries.mean() - global_timeseries.std() * global_cutoff))

        frame_diff = np.mean(np.abs(np.diff(depvars_timeseries, axis=0)), axis = 1)
        diff_spikes = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * difference_cutoff),
                                    np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * difference_cutoff))
        # build spike regressors
        spikes = pd.DataFrame([x + 1 for x in range(len(global_timeseries))], columns=["TR"])
        for i, loc in enumerate(global_spikes):
            spikes["spike_global" + str(i + 1)] = 0
            spikes.loc[int(loc),"spike_global" + str(i + 1)] = 1

        # build FD regressors
        for i, loc in enumerate(diff_spikes):
            spikes["spike_diff" + str(i + 1)] = 0
            spikes.loc[int(loc),"spike_diff" + str(i + 1)] = 1

        spikes.drop(columns = "TR", inplace=True)

        if old_confounds_df is not None:
            confounds = pd.concat((old_confounds_df.reset_index(drop=True), spikes.reset_index(drop=True)), axis = 1)
        else:
            confounds = spikes
        
        return confounds, 