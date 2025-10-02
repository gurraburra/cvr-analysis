#%%
import numpy as np
from process_control import ProcessWorkflow, ValueNode, CustomNode, ConditionalNode, IteratingNode
from nilearn.maskers import NiftiMasker
import nibabel as nib
from cvr_analysis.default.dev.c
lasses.regress import RegressNilearn
from cvr_analysis.default.dev.c
lasses.xcorr import XCorr
from cvr_analysis.default.dev.c
lasses.find_peak import FindPeaks, FilterPeaks

# %% No delay map
zero_delay = CustomNode(lambda dv_ts : (np.full(dv_ts.shape[1], 0.0), ), ("delay", ))
# down_sample
down_sample = CustomNode(lambda arr, down_sampling_factor : (arr[::down_sampling_factor], ), ("down_sampled_arr"))
# regress
regress_node = RegressNilearn(description="regress")
# workflow
no_delay_regress_wf = ProcessWorkflow(
    (
        # down_sample
        (ProcessWorkflow.input.dv_ts, down_sample.input.arr),
        (ProcessWorkflow.input.down_sampling_factor, down_sample.input.down_sampling_factor),
        # delay
        (down_sample.output.down_sampled_arr, zero_delay.input.dv_ts),
        (zero_delay.output.delay, ProcessWorkflow.output.delay),
        # regress
        (ProcessWorkflow.input._, regress_node.input['regressor_ts', 'confound_regressor_correlation_threshold', 'confounds_df', 'noise_model']),
        (ProcessWorkflow.input.down_sampling_factor, regress_node.input.regressor_down_sampling_factor),
        (down_sample.output.down_sampled_arr, regress_node.input.dv_ts),
        (ValueNode(0.0).output.value, regress_node.input.regressor_delay),
        (ValueNode(1).output.value, regress_node.input.regressor_time_step),
        (regress_node.output.all - regress_node.output["betas", "design_matrix"], ProcessWorkflow.output._),
        # sim metric
        (ValueNode(None).output.value, ProcessWorkflow.output.similarity_timeshifts),
        (ValueNode(None).output.value, ProcessWorkflow.output.similarity_metric),
    ), description="no delay regress"
)
# %% find optimal similiarity given metric
# maximum
def getMaximum(similarity_measure, bipolar = True):
    if bipolar:
        sim = similarity_measure
    else:
        sim = np.abs(similarity_measure)
    return np.nanargmax(sim, axis = 0), 

get_maximum = CustomNode(getMaximum, ("indices",))

# peaks
find_peaks = FindPeaks()
filter_peaks = FilterPeaks()


# workflow
peak_wf = ProcessWorkflow(
    (
        # find peaks
        (ProcessWorkflow.input.similarity_measure, find_peaks.input.y),
        (ProcessWorkflow.input.bipolar, find_peaks.input.bipolar),
        (ProcessWorkflow.input.min_peak_height, find_peaks.input.min_height),
        (ProcessWorkflow.input.max_nr_delay_peaks, find_peaks.input.max_nr_peaks),
        # filter peaks
        (ProcessWorkflow.input._, filter_peaks.input.timeseries_masker),
        (ProcessWorkflow.input.filter_peaks_type, filter_peaks.input.filter_type),
        (ProcessWorkflow.input.filter_peaks_kernel, filter_peaks.input.kernel_size),
        (find_peaks.output.peaks_array, filter_peaks.input.peaks_array),
        (filter_peaks.output.filtered_peaks, ProcessWorkflow.output.indices)
    )
)
# conditional node
optimal_similarity_cond = ConditionalNode("optimal_similarity_strategy", 
                                 {"maximum" : get_maximum, 
                                  "absolute_maximum" : get_maximum, 
                                  "peak" : peak_wf, 
                                  "absolute_peak" : peak_wf},
                                    default_condition="absolute_peak", 
                                        output_mapping={"delay_indices" : (get_maximum.output.indices, peak_wf.output.indices)})
# infer bipolar metric
bipolar_metric = CustomNode(lambda optimal_similarity_strategy : False if optimal_similarity_strategy is None else "absolute" in optimal_similarity_strategy, ("bipolar", ))

# pick indices
# workflow
find_optimal_sim_wf = ProcessWorkflow(
    (
        # bipolar
        (ProcessWorkflow.input._, bipolar_metric.input.optimal_similarity_strategy),
        # optimal similarity cond
        (ProcessWorkflow.input._, optimal_similarity_cond.input.all - optimal_similarity_cond.input.bipolar),
        (bipolar_metric.output.bipolar, optimal_similarity_cond.input.bipolar),
        (optimal_similarity_cond.output.all, ProcessWorkflow.output._)
    )
)

# %% xcorr delay regress
# xcorr
xcorr_node = XCorr(description="cross corr")
# group delays and dv_ts
def split2DArray(arr : np.ndarray, labels : np.ndarray):
    # assert dimensions
    assert arr.ndim == 2, "'arr' needs to a 2D array"
    assert labels.ndim == 1, "'labels' needs to a 1D array"
    assert arr.shape[1] == labels.shape[0], "Number of columns in 'arr' needs to match length of 'labels"
    # split
    unq_labels = np.unique(labels)
    split_arrays = [arr[:, labels == label] for label in unq_labels]
    return split_arrays, unq_labels
split_2d_array = CustomNode(split2DArray, ("array_list","list_labels"))
# reorder regress output
def reorderRegressOutput(labels : np.ndarray, r2 : np.array, adj_r2 : np.array, regr_beta : np.array, regr_se : np.array, regr_t : np.array, regr_p : np.array, pred_T : np.ndarray, dof : np.array):
    # assert dimensions
    arrs = (r2, adj_r2, regr_beta, regr_se, regr_t, regr_p, pred_T)
    # regroup
    reorder_arrs = tuple(np.empty(arr.shape) for arr in arrs)
    dof_arr = np.empty(labels.shape)
    # map back
    idx = 0
    for i,label in enumerate(np.unique(labels)):
        mask = labels == label
        nr_elem = np.sum(mask)
        for arr, reorder_arr in zip(arrs, reorder_arrs):
            reorder_arr[mask] = arr[idx : idx + nr_elem]
        # handle dof special
        dof_arr[mask] = np.full(nr_elem, dof[i])
        idx += nr_elem

    # pivot prediction and add dof
    return reorder_arrs[:-1] + (reorder_arrs[-1].T, dof_arr,)
reorder_regress_output = CustomNode(reorderRegressOutput, ("r_squared", "adjusted_r_squared", "regressor_beta", "regressor_se", "regressor_t", "regressor_p", "predictions", "dof"), description="reorder regress outputs")
# iterate regress
regress_pivot_pred = ProcessWorkflow(
    (
        (ProcessWorkflow.input._, regress_node.input.all),
        (regress_node.output.all - regress_node.output.predictions, ProcessWorkflow.output._),
        (regress_node.output.predictions.T, ProcessWorkflow.output.predictions_T)
    )
)
iterate_regress_xcorr = IteratingNode(regress_pivot_pred, ("dv_ts", "regressor_delay"), "regress", ("design_matrix", "betas"), description="X-corr")
# indices to timeshift
indices_to_timeshift = CustomNode(lambda indices, timeshifts : (timeshifts[indices],), ("timeshift", ))
# xcorr find delay workflow
xcorr_delay_wf = ProcessWorkflow(
    (
        # xcorr
        (ProcessWorkflow.input._, xcorr_node.input.all),
        (xcorr_node.output.all, ProcessWorkflow.output._),
        # delay strategy
        (ProcessWorkflow.input._, find_optimal_sim_wf.input.all - find_optimal_sim_wf.input.similarity_measure),
        (xcorr_node.output.correlations, find_optimal_sim_wf.input.similarity_measure),
        (find_optimal_sim_wf.output.delay_indices, ProcessWorkflow.output.delay_indices),
        # indices to timeshift
        (xcorr_node.output.timeshifts, indices_to_timeshift.input.timeshifts),
        (find_optimal_sim_wf.output.delay_indices, indices_to_timeshift.input.indices),
        (indices_to_timeshift.output.timeshift, ProcessWorkflow.output.delay),
    ), description="xcorr delay"
)
xcorr_delay_regress_wf = ProcessWorkflow(
    (
        # xcorr delay
        (ProcessWorkflow.input.dv_ts, xcorr_delay_wf.input.signals),
        (ProcessWorkflow.input.regressor_ts, xcorr_delay_wf.input.reference),
        (ProcessWorkflow.input.time_step, xcorr_delay_wf.input.time_step),
        (ProcessWorkflow.input.xcorr_phat, xcorr_delay_wf.input.phat),
        (ProcessWorkflow.input.xcorr_window, xcorr_delay_wf.input.window),
        (ProcessWorkflow.input.delay_lower_limit, xcorr_delay_wf.input.lower_limit),
        (ProcessWorkflow.input.delay_upper_limit, xcorr_delay_wf.input.upper_limit),
        (ProcessWorkflow.input.timeseries_masker, xcorr_delay_wf.input.timeseries_masker),
        (ProcessWorkflow.input.optimal_similarity_strategy, xcorr_delay_wf.input.optimal_similarity_strategy),
        (ProcessWorkflow.input.min_peak_height, xcorr_delay_wf.input.min_peak_height),
        (ProcessWorkflow.input.max_nr_delay_peaks, xcorr_delay_wf.input.max_nr_delay_peaks),
        (ProcessWorkflow.input.filter_peaks_type, xcorr_delay_wf.input.filter_peaks_type),
        (ProcessWorkflow.input.filter_peaks_kernel, xcorr_delay_wf.input.filter_peaks_kernel),
        (xcorr_delay_wf.output.timeshifts, ProcessWorkflow.output.similarity_timeshifts),
        (xcorr_delay_wf.output.correlations, ProcessWorkflow.output.similarity_metric),
        (xcorr_delay_wf.output.delay, ProcessWorkflow.output.delay),
        # down_sample
        (ProcessWorkflow.input.dv_ts, down_sample.input.arr),
        (ProcessWorkflow.input.down_sampling_factor, down_sample.input.down_sampling_factor),
        # split array according to delays
        (down_sample.output.down_sampled_arr, split_2d_array.input.arr),
        (xcorr_delay_wf.output.delay, split_2d_array.input.labels),
        # iterate regress
        (ProcessWorkflow.input._, iterate_regress_xcorr.input['regressor_ts', 'confound_regressor_correlation_threshold', 'confounds_df', 'noise_model']),
        (ProcessWorkflow.input.nr_processes, iterate_regress_xcorr.input.regressIter_nr_parallel_processes),
        (ProcessWorkflow.input.time_step, iterate_regress_xcorr.input.regressor_time_step),
        (ProcessWorkflow.input.down_sampling_factor, iterate_regress_xcorr.input.regressor_down_sampling_factor),
        (ProcessWorkflow.input.show_pbar, iterate_regress_xcorr.input.regressIter_show_pbar),
        (ValueNode(True).output.value, iterate_regress_xcorr.input.regressIter_concat_array_outputs),
        (split_2d_array.output.array_list, iterate_regress_xcorr.input.regressIter_dv_ts),
        (split_2d_array.output.list_labels, iterate_regress_xcorr.input.regressIter_regressor_delay),
        # reorder output
        (xcorr_delay_wf.output.delay, reorder_regress_output.input.labels),
        (iterate_regress_xcorr.output.regressIter_adjusted_r_squared, reorder_regress_output.input.adj_r2),
        (iterate_regress_xcorr.output.regressIter_dof, reorder_regress_output.input.dof),
        (iterate_regress_xcorr.output.regressIter_predictions_T, reorder_regress_output.input.pred_T),
        (iterate_regress_xcorr.output.regressIter_r_squared, reorder_regress_output.input.r2),
        (iterate_regress_xcorr.output.regressIter_regressor_beta, reorder_regress_output.input.regr_beta),
        (iterate_regress_xcorr.output.regressIter_regressor_p, reorder_regress_output.input.regr_p),
        (iterate_regress_xcorr.output.regressIter_regressor_se, reorder_regress_output.input.regr_se),
        (iterate_regress_xcorr.output.regressIter_regressor_t, reorder_regress_output.input.regr_t),
        (reorder_regress_output.output.all, ProcessWorkflow.output._)
    ), description="xcorr delay regress"
)
 
# %% R2 delay regress
# create delay timevector
def _createDelay(dv_ts, regressor_ts, time_step, lower_limit = None, upper_limit = None):
    # timeshifts
    timeshifts = np.arange(-regressor_ts.shape[0]+1, dv_ts.shape[0], 1) * time_step
    # mask
    mask = np.full_like(timeshifts, True, dtype = bool)
    if lower_limit is not None:
        mask[timeshifts < lower_limit] = False
    if upper_limit is not None:
        mask[timeshifts > upper_limit] = False
    # check mask
    if not np.any(mask):
        if lower_limit is not None and upper_limit is not None:
            # pick value closet to the limits
            mask[np.argmin(np.abs(timeshifts - (lower_limit + upper_limit) / 2))] = True
        else:
            raise ValueError("Incorrect limits specified.")
    # mask timeshifts
    timeshifts_masked = timeshifts[mask]
    return timeshifts_masked,
create_delay_node = CustomNode(_createDelay, ("timeshifts",))
# iterate regress r2
iterate_regress_r2 = IteratingNode(regress_pivot_pred, ("regressor_delay",), "regress", ("design_matrix", "betas"), description="R2")
# pick specific index from computed regress output
def _pickIndex(indices, iter_adjusted_r_squared, iter_dof, iter_predictions_T, iter_r_squared, iter_regressor_beta, iter_regressor_p, iter_regressor_se, iter_regressor_t):
    voxel_arr = np.arange(iter_adjusted_r_squared.shape[1])
    return iter_adjusted_r_squared[indices, voxel_arr], iter_dof[indices], iter_predictions_T[indices, voxel_arr].T, iter_r_squared[indices, voxel_arr], iter_regressor_beta[indices, voxel_arr], iter_regressor_p[indices, voxel_arr], iter_regressor_se[indices, voxel_arr], iter_regressor_t[indices, voxel_arr]
pick_index_node = CustomNode(_pickIndex, ("adjusted_r_squared", "dof", "predictions", "r_squared", "regressor_beta", "regressor_p", "regressor_se", "regressor_t"))
# Workflow
r2_regress_delay_wf = ProcessWorkflow(
    (
        # timeshifts
        (ProcessWorkflow.input.dv_ts, create_delay_node.input.dv_ts),
        (ProcessWorkflow.input.regressor_ts, create_delay_node.input.regressor_ts),
        (ProcessWorkflow.input.time_step, create_delay_node.input.time_step),
        (ProcessWorkflow.input.delay_lower_limit, create_delay_node.input.lower_limit),
        (ProcessWorkflow.input.delay_upper_limit, create_delay_node.input.upper_limit),
        (create_delay_node.output.timeshifts, ProcessWorkflow.output.similarity_timeshifts),
        # down_sample
        (ProcessWorkflow.input.dv_ts, down_sample.input.arr),
        (ProcessWorkflow.input.down_sampling_factor, down_sample.input.down_sampling_factor),
        # iterate regress
        (ProcessWorkflow.input._, iterate_regress_r2.input['regressor_ts', 'confound_regressor_correlation_threshold', 'confounds_df', 'noise_model']),
        (ProcessWorkflow.input.nr_processes, iterate_regress_r2.input.regressIter_nr_parallel_processes),
        (ProcessWorkflow.input.time_step, iterate_regress_r2.input.regressor_time_step),
        (ProcessWorkflow.input.down_sampling_factor, iterate_regress_r2.input.regressor_down_sampling_factor),
        (ProcessWorkflow.input.show_pbar, iterate_regress_r2.input.regressIter_show_pbar),
        (ValueNode(False).output.value, iterate_regress_r2.input.regressIter_concat_array_outputs),
        (down_sample.output.down_sampled_arr, iterate_regress_r2.input.dv_ts),
        (create_delay_node.output.timeshifts, iterate_regress_r2.input.regressIter_regressor_delay),
        (iterate_regress_r2.output.regressIter_r_squared, ProcessWorkflow.output.similarity_metric),
        # delay strategy
        (ProcessWorkflow.input._, optimal_similarity_cond.input["optimal_similarity_strategy", "timeseries_masker", "filter_peaks_kernel", "filter_peaks_type", "min_peak_height", "max_nr_delay_peaks"]),
        (ValueNode(False).output.value, optimal_similarity_cond.input.bipolar),
        (iterate_regress_r2.output.regressIter_r_squared, optimal_similarity_cond.input.similarity_measure),
        # indices to timeshift
        (optimal_similarity_cond.output.delay_indices, indices_to_timeshift.input.indices),
        (create_delay_node.output.timeshifts, indices_to_timeshift.input.timeshifts),
        (indices_to_timeshift.output.timeshift, ProcessWorkflow.output.delay),
        # pick index
        (optimal_similarity_cond.output.delay_indices, pick_index_node.input.indices),
        (iterate_regress_r2.output.regressIter_adjusted_r_squared, pick_index_node.input.iter_adjusted_r_squared),
        (iterate_regress_r2.output.regressIter_dof, pick_index_node.input.iter_dof),
        (iterate_regress_r2.output.regressIter_predictions_T, pick_index_node.input.iter_predictions_T),
        (iterate_regress_r2.output.regressIter_r_squared, pick_index_node.input.iter_r_squared),
        (iterate_regress_r2.output.regressIter_regressor_beta, pick_index_node.input.iter_regressor_beta),
        (iterate_regress_r2.output.regressIter_regressor_p, pick_index_node.input.iter_regressor_p),
        (iterate_regress_r2.output.regressIter_regressor_se, pick_index_node.input.iter_regressor_se),
        (iterate_regress_r2.output.regressIter_regressor_t, pick_index_node.input.iter_regressor_t),
        (pick_index_node.output.all, ProcessWorkflow.output._),
    ), description="R2 regresss delay"
)
# %% delay regress cond
delay_regress_cond = ConditionalNode("delay_strategy",
                                     {None : no_delay_regress_wf,
                                      "xcorr" : xcorr_delay_regress_wf,
                                      "r2" : r2_regress_delay_wf}, default_condition=None)
# %%
# test
if __name__ == "__main__":
    tm = NiftiMasker()
    img = nib.Nifti1Image(np.random.rand(10,10,50,200)*5, np.eye(4))
    data = tm.fit_transform(img)
    xcorr_node.cache_data = True
    optimal_similarity_cond.cache_data = True
    find_peaks.cache_data = True
    peak_wf.cache_data = True
    filter_peaks.cache_data = True

    r = delay_regress_cond.run(delay_strat = "xcorr", timeseries_ma = tm, dv_ts = data, regressor_ts = data[:,0], delay_lower = -10, delay_upper = 10, optimal_sim = "maximum", update_cache=True, verbose = True, nr_processes = 4)

# %%
