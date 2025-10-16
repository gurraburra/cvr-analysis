# %%
import numpy as np
from nilearn.masking import compute_epi_mask, apply_mask

# process control
from process_control import *
# custom packages
from cvr_analysis.default.helpers.workflows.load_regressor_wf import load_regressor_wf
from cvr_analysis.default.helpers.classes.load_in_data import LoadBOLDData, LoadBidsImg, CropBOLDImg, VoxelTimeSeriesMasker, RoiTimeSeriesMasker

# %%

##########################################################################################################################################
# post-processing wf
##########################################################################################################################################

############################################################################################
# data loader wf
############################################################################################

##############################################
# bold data loader
##############################################
bold_loader = LoadBOLDData(description="load in bold data")

##############################################
# regressor data loader (imported)
##############################################

##############################################
# voxel mask loader
##############################################
load_mask = LoadBidsImg(description="load in data mask")
compute_epi = CustomNode(lambda bold_img : compute_epi_mask(bold_img), outputs="voxel_mask_img", description="nilearn compute EPI")
# either load existing mask or compute new mask
voxel_mask_loader = ConditionalNode("voxel_mask",
                                    condition_node_map={
                                        ConditionalNode.no_match_condition : load_mask,
                                        "nlEPI" : compute_epi,
                                    },
                                    input_mapping={"voxel_mask" : load_mask.input.desc},
                                    output_mapping={"voxel_mask_img" : load_mask.output.bids_img}, 
                                    description="load voxel mask")
# crop bold img
crop_bold_img = CropBOLDImg(description="crop bold data img")

##############################################
# timeseries masker
##############################################
# voxel data
voxel_masker = VoxelTimeSeriesMasker(description="create voxel timeseries masker")
# roi data
label_masker_loader = LoadBidsImg(description="load labels img")
roi_masker = RoiTimeSeriesMasker(description="create roi timeseries masker")
load_create_roi_masker = ProcessWorkflow(
    (
        # labels masker loader
        (ProcessWorkflow.input._, label_masker_loader.input.all / label_masker_loader.input.suffix / label_masker_loader.input.data_type),
        (ValueNode("dseg").output.value, label_masker_loader.input.suffix),
        (ValueNode("func").output.value, label_masker_loader.input.data_type),
        # create timeseries masker
        (ProcessWorkflow.input._, roi_masker.input[("voxel_mask_img", "spatial_smoothing_fwhm")]),
        (label_masker_loader.output.bids_img, roi_masker.input.labels_img),
        (roi_masker.output.roi_masker, ProcessWorkflow.output._),
    ),
    description="load and create roi timeseries masker"     
)

# conditioanlly select timeseries masker -> None : use voxels as timeseries, other use a dseg labels masker
timeseries_masker = ConditionalNode("roi_masker", default_condition=None, 
                                            condition_node_map={
                                                                    None : voxel_masker,
                                                                    ConditionalNode.no_match_condition : load_create_roi_masker
                                                                },
                                            input_mapping={"roi_masker" : load_create_roi_masker.input.desc},
                                            output_mapping={"timeseries_masker" : (voxel_masker.output.voxel_masker, 
                                                                                    load_create_roi_masker.output.roi_masker)},
                                            description="timeseries masker"
                                        )
# get timeseries
get_bold_timeseries = CustomNode(lambda bold_img, timeseries_masker : timeseries_masker.fit_transform(bold_img), outputs="bold_timeseries", description="get bold timeseries")
generate_bold_times = CustomNode(lambda tr, nr_measurements : np.arange(nr_measurements) * tr, outputs="bold_times", description="generate bold timeseries times")

# compute global signal for alignment
def globalSignal(bold_img, voxel_mask):
    ts = apply_mask(bold_img, voxel_mask)
    return np.mean(ts, axis = 1), 
comp_global_signal = CustomNode(globalSignal, outputs=("global_timeseries", ))

# %%
load_data_wf = ProcessWorkflow(
    (
        # bold_data
        (ProcessWorkflow.input._, bold_loader.input.all / bold_loader.input[("load_confounds",)]),
        (ProcessWorkflow.input.include_motion_confounds, bold_loader.input.load_confounds),
        (bold_loader.output.confounds_df, ProcessWorkflow.output.confounds_df),
        (bold_loader.output.tr,  ProcessWorkflow.output.bold_tr),
        (bold_loader.output.nr_measurements,  ProcessWorkflow.output.nr_measurements),
        # voxel mask loader
        (ProcessWorkflow.input._, voxel_mask_loader.input.all / voxel_mask_loader.input[("suffix", "bold_img", "data_type")]),
        (ValueNode("mask").output.value, voxel_mask_loader.input.suffix),
        (ValueNode("func").output.value, voxel_mask_loader.input.data_type),
        (bold_loader.output.bold_img, voxel_mask_loader.input.bold_img),
        (voxel_mask_loader.output.voxel_mask_img, ProcessWorkflow.output.voxel_mask_img),
        # crop bold data
        (voxel_mask_loader.output.voxel_mask_img, crop_bold_img.input.voxel_mask_img),
        (bold_loader.output.bold_img, crop_bold_img.input.bold_img),
        # timeseries masker
        (ProcessWorkflow.input._, timeseries_masker.input.all / timeseries_masker.input.voxel_mask_img),
        (crop_bold_img.output.resampled_voxel_mask_img, timeseries_masker.input.voxel_mask_img),
        (timeseries_masker.output.timeseries_masker, ProcessWorkflow.output._),
        # get timeseries
        (crop_bold_img.output.cropped_bold_img, get_bold_timeseries.input.bold_img),
        (timeseries_masker.output.timeseries_masker, get_bold_timeseries.input.timeseries_masker),
        (get_bold_timeseries.output.bold_timeseries, ProcessWorkflow.output.bold_timeseries),
        # compute global signal
        (crop_bold_img.output.resampled_voxel_mask_img, comp_global_signal.input.voxel_mask),
        (crop_bold_img.output.cropped_bold_img, comp_global_signal.input.bold_img),
        (comp_global_signal.output.global_timeseries, ProcessWorkflow.output._),
        # generate_bold_times
        (bold_loader.output.tr, generate_bold_times.input.tr),
        (bold_loader.output.nr_measurements, generate_bold_times.input.nr_measurements),
        (generate_bold_times.output.bold_times, ProcessWorkflow.output.bold_times),
        # regressor wf
        (ProcessWorkflow.input._, load_regressor_wf.input.all - load_regressor_wf.input[("global_times", "global_timeseries", "global_units", "data_type")]),
        (ValueNode("BOLD").output.value, (load_regressor_wf.input.global_units, ProcessWorkflow.output.bold_units)),
        (ValueNode("func").output.value, load_regressor_wf.input.data_type),
        (generate_bold_times.output.bold_times, load_regressor_wf.input.global_times),
        (comp_global_signal.output.global_timeseries, load_regressor_wf.input.global_timeseries),
        (load_regressor_wf.output.regressor_times, ProcessWorkflow.output.regressor_times),
        (load_regressor_wf.output.regressor_timeseries, ProcessWorkflow.output.regressor_timeseries),
        (load_regressor_wf.output.regressor_units, ProcessWorkflow.output.regressor_units),
    ),
    description="data loader workflow"
).setDefaultInputs(include_motion_confounds = True)
# %%