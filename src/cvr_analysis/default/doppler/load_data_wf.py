
# %%
import numpy as np
from nilearn.masking import compute_epi_mask

# process control
from process_control import *
# custom packages
from cvr_analysis.default.helpers.workflows.load_regressor_wf import load_regressor_wf
from cvr_analysis.default.helpers.classes.load_in_data import LoadDopplerData, LoadTimeseriesEvent, LoadPhysioData
from cvr_analysis.default.helpers.classes.signal_processing import NewSampleTime, ResampleTimeSeries, DetrendTimeSeries, TemporalFilterTimeSeries, TimeLimitTimeSeries
from cvr_analysis.default.helpers.classes.data_computation import Correlate, AlignTimeSeries

# %%

############################################################################################
# data loader wf
############################################################################################

##############################################
# doppler data loader
##############################################
doppler_loader = LoadPhysioData(description="load in doppler data")

##############################################
# regressor data loader (imported)
##############################################
# select regressor
# compute global signal for alignment
def globalSignal(doppler_ts : np.ndarray):
    if doppler_ts.ndim == 2:
        return doppler_ts.mean(axis = 1), 
    elif doppler_ts.ndim == 1:
        return doppler_ts, 
    else:
        raise ValueError("'doppler_ts' must be 1 or 2 dimensional")
comp_global_signal = CustomNode(globalSignal, outputs=("global_timeseries", ))

# check doppler units
def checkUnits(units):
    if isinstance(units, str):
        return units,
    else:
        assert len(set(units)) == 1, "doppler signals need to be all of same unit"
        return units[0],
check_units = CustomNode(checkUnits, outputs=("units", ))
# %%
load_data_wf = ProcessWorkflow(
    (
        # doppler_data
        (ProcessWorkflow.input._, doppler_loader.input.all - doppler_loader.input[("variables", "data_type")]),
        (ProcessWorkflow.input.doppler_variables, doppler_loader.input.variables),
        (ValueNode("doppler").output.value, doppler_loader.input.data_type),
        (doppler_loader.output.times, ProcessWorkflow.output.doppler_times),
        (doppler_loader.output.time_step,  ProcessWorkflow.output.doppler_tr),
        (doppler_loader.output.timeseries, ProcessWorkflow.output.doppler_timeseries),
        (doppler_loader.output.variables, ProcessWorkflow.output.doppler_variables),
        # check units
        (doppler_loader.output.units, check_units.input.units),
        (check_units.output.units, ProcessWorkflow.output.doppler_units),
        # compute global signal
        (doppler_loader.output.timeseries, comp_global_signal.input.doppler_ts),
        (comp_global_signal.output.global_timeseries, ProcessWorkflow.output._),
        # regressor wf
        (ProcessWorkflow.input._, load_regressor_wf.input.all - load_regressor_wf.input[("global_times", "global_timeseries", "global_units", "data_type")]),
        (ValueNode("doppler").output.value, load_regressor_wf.input.data_type),
        (doppler_loader.output.times, load_regressor_wf.input.global_times),
        (comp_global_signal.output.global_timeseries, load_regressor_wf.input.global_timeseries),
        (check_units.output.units, load_regressor_wf.input.global_units),
        (load_regressor_wf.output.regressor_times, ProcessWorkflow.output.regressor_times),
        (load_regressor_wf.output.regressor_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (load_regressor_wf.output.regressor_units, ProcessWorkflow.output.regressor_units),
    ),
    description="data loader workflow"
)