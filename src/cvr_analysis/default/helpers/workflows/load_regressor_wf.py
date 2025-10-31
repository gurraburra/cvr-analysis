#%%
# # process control
from process_control import *
# custom packages
from cvr_analysis.default.helpers.classes.load_in_data import LoadTimeseriesEvent, LoadPhysioData

__all__ = ["load_regressor_wf"]

##############################################
# regressor data loader
##############################################
# regressor type
def regressorType(regressor : str):
    if regressor == "global-signal":
        return "global-signal", 
    elif regressor.startswith("physio:"):
        return "physio", 
    else:
        return "event", 
regressor_type = CustomNode(regressorType, outputs=("regressor_type",))
# split physio name
def splitPhysioName(regressor : str):
    # regressor in the form: physio:<recording>:<variable>:<variable>
    physio_parts = regressor.split(":")
    if len(physio_parts) > 2:
        recording = None if physio_parts[1] == "None" else physio_parts[1]
        return recording, physio_parts[2:]
    else:
        raise ValueError(f"incorrect defined physiological recording '{recording}'")
physio_name_split = CustomNode(splitPhysioName, outputs=("recording", "variables"))
# load physio
physio_loader = LoadPhysioData()
# assert single variables
def singlePhysVar(variables):
    if len(variables) > 1:
        raise ValueError(f"multiple variables specified for regressor")
    else:
        return variables[0],
single_phys_var = CustomNode(singlePhysVar, outputs=("variable",))
physio_wf = ProcessWorkflow(
    (
        # split physio name
        (ProcessWorkflow.input._, physio_name_split.input.regressor),
        # single var
        (physio_name_split.output.variables, single_phys_var.input.variables),
        # physio loader
        (ProcessWorkflow.input._, physio_loader.input.all - physio_loader.input[("recording", "variables")]),
        (physio_name_split.output.recording, physio_loader.input.recording),
        (single_phys_var.output.variable, physio_loader.input.variables),
        (physio_loader.output.all / physio_loader.output.units, ProcessWorkflow.output._),
        (physio_loader.output.units, ProcessWorkflow.output.unit)
    ), description="load physio data"
)
event_loader = LoadTimeseriesEvent()

# global signal
pass_global_regressor = ProcessWorkflow(
    (
        (ProcessWorkflow.input.global_times, ProcessWorkflow.output.regressor_times),
        (ProcessWorkflow.input.global_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (ProcessWorkflow.input.global_unit, ProcessWorkflow.output.regressor_unit),
    )
)
#%%
# loader
conditional_loader = ConditionalNode("regressor_type", 
                                  {"global-signal" : pass_global_regressor, 
                                   "physio" : physio_wf,
                                   "event" : event_loader}, 
                                   input_mapping =
                                   {
                                       "regressor" : event_loader.input.event_name,
                                   },
                                   output_mapping = 
                                   {
                                       "regressor_times" : (event_loader.output.times,physio_wf.output.times),
                                       "regressor_timeseries" : (event_loader.output.timeseries,physio_wf.output.timeseries), 
                                       "regressor_unit" : (event_loader.output.unit,physio_wf.output.unit) 
                                   },
                                   description="conditionally load regressor data").setDefaultInputs(regressor = ProcessNode.no_default_input)

# convert time unit
def convFactorUnit(timeseries, unit, use_mmhg = True):
    if timeseries is not None:
        if unit.lower() == 'kpa':
            if use_mmhg:
                return timeseries*7.50061683, "mmHg"
            else:
                return timeseries, "kPa"
        elif unit.lower() == 'mmhg':
            if not use_mmhg:
                return 0.133322368, "kPa"
            else:
                return timeseries, "mmHg"
        else:
            return timeseries, unit
    else:
        return None, None
conv_factor = CustomNode(convFactorUnit, outputs=("conv_timeseries", "conv_unit"))

# load regressor wf
load_regressor_wf = ProcessWorkflow(
    (
        # regressor type
        (ProcessWorkflow.input._, regressor_type.input.regressor),
        # conditional regressor ts loader
        (ProcessWorkflow.input._, conditional_loader.input.all / conditional_loader.input[("regressor_type")]),
        # (ProcessWorkflow.input.regressor, conditional_loader.input.event_name),
        (regressor_type.output.regressor_type, conditional_loader.input.regressor_type),
        (conditional_loader.output.regressor_times, ProcessWorkflow.output._),
        # convert unit
        (ProcessWorkflow.input._, conv_factor.input.use_mmhg),
        (conditional_loader.output.regressor_timeseries, conv_factor.input.timeseries),
        (conditional_loader.output.regressor_unit, conv_factor.input.unit),
        (conv_factor.output.conv_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (conv_factor.output.conv_unit, ProcessWorkflow.output.regressor_unit),
    ), description="regressor wf"
)
# %%