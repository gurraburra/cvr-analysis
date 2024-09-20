from process_control import ProcessNode
import pandas as pd
import numpy as np
from nilearn import glm


class MotionConfounds(ProcessNode):
    outputs = ("motion_confounds_df", )
    
    def _run(self, confounds_df : pd.DataFrame, derivatives : bool = True, powers : bool = True) -> tuple:
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

        return confound_selected, 