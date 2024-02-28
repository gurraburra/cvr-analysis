from process_control import ProcessNode
import pandas as pd
import numpy as np
from nilearn import glm


class MotionConfounds(ProcessNode):
    outputs = ("confounds_df", )
    
    def _run(self, confounds_df : pd.DataFrame, derivatives : bool = True, powers : bool = True, standardize : bool = True, old_confounds_df : pd.DataFrame = None) -> tuple:
        confound_names = []
        include = [""]
        if derivatives:
            include.append("_derivative1")
        if powers:
            include.append("_power2")
        if powers and derivatives:
            include.append("_derivative1_power2")

        c_base = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
        for c in c_base:
            for i in include:
                confound_names.append(c + i)

        confound_selected = confounds_df.loc[:,confound_names]
        if standardize:
            confound_selected = (confound_selected - confound_selected.mean()) / confound_selected.std()

        if old_confounds_df is not None:
            confound_selected = pd.concat((old_confounds_df.reset_index(drop=True), confound_selected.reset_index(drop=True)), axis = 1)

        return confound_selected, 

class DriftConfounds(ProcessNode):
    outputs = ("confounds_df", )
    
    def _run(self, tr : float, nr_measurements : int, ref_slice : float = 0.5, drift_model : str = 'polynomial', drift_order : int = 1, high_pass : float = 0.01, old_confounds_df : pd.DataFrame = None) -> tuple:
        start_time = tr * ref_slice
        end_time = (nr_measurements - 1 + ref_slice) * tr
        frame_times = np.linspace(start_time, end_time, nr_measurements)

        drift_df = glm.first_level.make_first_level_design_matrix(frame_times, drift_model = drift_model, drift_order = drift_order, high_pass = high_pass)

        if old_confounds_df is not None:
            drift_df = pd.concat((old_confounds_df.reset_index(drop=True), drift_df.reset_index(drop=True)), axis = 1)

        return drift_df, 

class SpikeConfounds(ProcessNode):
    outputs = ("confounds_df", )
    
    def _run(self, bold_data : np.ndarray, global_cutoff : float = 3, difference_cutoff : float = 3, old_confounds_df : pd.DataFrame = None) -> tuple:
        global_mean = bold_data.mean(axis = 1)
        global_spikes = np.append(np.where(global_mean > global_mean.mean() + global_mean.std() * global_cutoff),
                                    np.where(global_mean < global_mean.mean() - global_mean.std() * global_cutoff))

        frame_diff = np.mean(np.abs(np.diff(bold_data, axis=0)), axis = 1)
        diff_spikes = np.append(np.where(frame_diff > np.mean(frame_diff) + np.std(frame_diff) * difference_cutoff),
                                    np.where(frame_diff < np.mean(frame_diff) - np.std(frame_diff) * difference_cutoff))
        # build spike regressors
        spikes = pd.DataFrame([x + 1 for x in range(len(global_mean))], columns=["TR"])
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