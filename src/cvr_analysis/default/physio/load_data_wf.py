
# %%
import numpy as np
from nilearn.masking import compute_epi_mask

# process control
from process_control import *
# custom packages
from cvr_analysis.default.helpers.workflows.load_regressor_wf import load_regressor_wf, conv_factor
from cvr_analysis.default.helpers.classes.load_in_data import LoadDopplerData, LoadTimeseriesEvent, LoadPhysioData
from cvr_analysis.default.helpers.classes.signal_processing import NewSampleTime, ResampleTimeSeries, DetrendTimeSeries, TemporalFilterTimeSeries, TimeLimitTimeSeries
from cvr_analysis.default.helpers.classes.data_computation import Correlate, AlignTimeSeries

# %%

############################################################################################
# data loader wf
############################################################################################

##############################################
# physio data loader
##############################################
physio_loader = LoadPhysioData(description="load in physio data")
# asssert variables is tuple
def variable2Tuple(variables):
    if variables is None:
        return None
    elif isinstance(variables, str):
        return (variables,), 
    elif isinstance(variables,(list,tuple)):
        return tuple(variables),
    else:
        raise ValueError("variables need to str, list or tuple")
var_2_tuple = CustomNode(variable2Tuple, outputs=("variables",), description="variables 2 tuple")
##############################################
# regressor data loader (imported)
##############################################
# select regressor
# compute global signal for alignment
def globalSignal(physio_ts : np.ndarray):
    if physio_ts.ndim == 2:
        return physio_ts.mean(axis = 1), 
    elif physio_ts.ndim == 1:
        return physio_ts, 
    else:
        raise ValueError("'physio_ts' must be 1 or 2 dimensional")
comp_global_signal = CustomNode(globalSignal, outputs=("global_timeseries", ))

# check physio units
def checkUnits(units):
    if isinstance(units, str):
        return units,
    else:
        assert len(set(units)) == 1, "physio signals need to be all of same unit"
        return units[0],
check_units = CustomNode(checkUnits, outputs=("unit", ))
# convert to mmHg
conv_physio = conv_factor.copy()
# workflow
load_data_wf = ProcessWorkflow(
    (
        # variables to tuple
        (ProcessWorkflow.input.physio_variables, var_2_tuple.input.variables),
        # physio_data
        (ProcessWorkflow.input._, physio_loader.input.all - physio_loader.input[("variables", "recording")]),
        (ProcessWorkflow.input.physio_recording, physio_loader.input.recording),
        (var_2_tuple.output.variables, physio_loader.input.variables),
        (physio_loader.output.times, ProcessWorkflow.output.physio_times),
        (physio_loader.output.time_step,  ProcessWorkflow.output.physio_tr),
        (physio_loader.output.variables, ProcessWorkflow.output.physio_variables),
        # check units
        (physio_loader.output.units, check_units.input.units),
        # convert unit
        (ProcessWorkflow.input._, conv_physio.input.use_mmhg),
        (physio_loader.output.timeseries, conv_physio.input.timeseries),
        (check_units.output.unit, conv_physio.input.unit),
        (conv_physio.output.conv_timeseries, ProcessWorkflow.output.physio_timeseries),
        (conv_physio.output.conv_unit, ProcessWorkflow.output.physio_unit),
        # compute global signal
        (conv_physio.output.conv_timeseries, comp_global_signal.input.physio_ts),
        (comp_global_signal.output.global_timeseries, ProcessWorkflow.output._),
        # regressor wf
        (ProcessWorkflow.input._, load_regressor_wf.input.all - load_regressor_wf.input[("global_times", "global_timeseries", "global_unit")]),
        (physio_loader.output.times, load_regressor_wf.input.global_times),
        (comp_global_signal.output.global_timeseries, load_regressor_wf.input.global_timeseries),
        (conv_physio.output.conv_unit, load_regressor_wf.input.global_unit),
        (load_regressor_wf.output.regressor_times, ProcessWorkflow.output.regressor_times),
        (load_regressor_wf.output.regressor_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (load_regressor_wf.output.regressor_unit, ProcessWorkflow.output.regressor_unit),
    ),
    description="data loader workflow"
).setDefaultInputs(data_type = ProcessNode.no_default_input)
# %%