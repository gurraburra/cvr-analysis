        
# %%
# process control
from process_control import *
from cvr_analysis.default.helpers.classes.signal_processing import MaskTimeSeries, DTW
from cvr_analysis.default.helpers.classes.data_computation import Correlate, AlignTimeSeries, FilterTimeshifts, PCAReducedTimeSeries, HistPeak
#%%
##############################################
# refine regressor
##############################################
##--##--##--##--##--##--##--##--##--##--##--##
# find timeshift
##--##--##--##--##--##--##--##--##--##--##--##
# iterate correlate over depvars timeseries
iterate_correlate_wf = IteratingNode(Correlate(), iterating_inputs="signal_timeseries_a", iterating_name="depvars", description="iterate correlate depvars timeseries").setDefaultInputs(depvarsIter_nr_parallel_processes = -1)
# filter timeshifts
filter_timeshifts = FilterTimeshifts(description="filter timeshifts")
# histogram peak
hist_peak = HistPeak(description="histogram peak")
# correlation bounds
# add if not None
add_none_lower = CustomNode(lambda x = None, y = None : x + y if x is not None and y is not None else None, description="add none")
add_none_upper = add_none_lower.copy()
# %%
# find timeshift wf
find_timeshift_wf = ProcessWorkflow(
    (
        # correlation bounds
        (ProcessWorkflow.input.align_regressor_lower_bound, add_none_lower.input.x),
        (ProcessWorkflow.input.align_regressor_upper_bound, add_none_upper.input.x),
        (ProcessWorkflow.input.reference_regressor_timeshift, add_none_lower.input.y),
        (ProcessWorkflow.input.reference_regressor_timeshift, add_none_upper.input.y),
        (add_none_lower.output.output, ProcessWorkflow.output.align_regressor_absolute_lower_bound),
        (add_none_upper.output.output, ProcessWorkflow.output.align_regressor_absolute_upper_bound),
        # iterate correlate
        (ProcessWorkflow.input.nr_parallel_processes, iterate_correlate_wf.input.depvarsIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar, iterate_correlate_wf.input.depvarsIter_show_pbar),
        (ProcessWorkflow.input.depvars_timeseries.T, iterate_correlate_wf.input.depvarsIter_signal_timeseries_a),
        (ProcessWorkflow.input.regressor_timeseries, iterate_correlate_wf.input.signal_timeseries_b),
        (ProcessWorkflow.input.sample_time, iterate_correlate_wf.input.time_step),
        (ProcessWorkflow.input.maxcorr_bipolar, iterate_correlate_wf.input.bipolar),
        (ProcessWorkflow.input.correlation_window, iterate_correlate_wf.input.window),
        (ProcessWorkflow.input.correlation_phat, iterate_correlate_wf.input.phat),
        (ProcessWorkflow.input.correlation_peak_threshold, iterate_correlate_wf.input.peak_threshold),
        (ProcessWorkflow.input.correlation_multi_peak_strategy, iterate_correlate_wf.input.multi_peak_strategy),
        (ProcessWorkflow.input.reference_regressor_timeshift, iterate_correlate_wf.input.ref_timeshift),
        (ValueNode(False).output.value, iterate_correlate_wf.input.depvarsIter_concat_array_outputs),
        (add_none_lower.output.output, iterate_correlate_wf.input.lower_limit),
        (add_none_upper.output.output, iterate_correlate_wf.input.upper_limit),
        (iterate_correlate_wf.output[("depvarsIter_timeshifts", "depvarsIter_correlations", "depvarsIter_fit_status")] , ProcessWorkflow.output._),
        # filter timeshift
        (ProcessWorkflow.input.timeseries_masker, filter_timeshifts.input.timeseries_masker),
        (ProcessWorkflow.input.filter_timeshifts_size, filter_timeshifts.input.size),
        (ProcessWorkflow.input.filter_timeshifts_filter_type, filter_timeshifts.input.filter_type),
        (ProcessWorkflow.input.filter_timeshifts_smooth_fwhm, filter_timeshifts.input.smooth_fwhm),
        (iterate_correlate_wf.output.depvarsIter_timeshift_maxcorr, filter_timeshifts.input.timeshift_maxcorr),
        (iterate_correlate_wf.output.depvarsIter_maxcorr, filter_timeshifts.input.maxcorr),
        (iterate_correlate_wf.output.depvarsIter_timeshifts, filter_timeshifts.input.timeshifts),
        (iterate_correlate_wf.output.depvarsIter_correlations, filter_timeshifts.input.correlations),
        (filter_timeshifts.output.filtered_timeshift_maxcorr, ProcessWorkflow.output.depvarsIter_timeshift_maxcorr),
        (iterate_correlate_wf.output.depvarsIter_fit_status, filter_timeshifts.input.fit_status),
        (filter_timeshifts.output.filtered_maxcorr, ProcessWorkflow.output.depvarsIter_maxcorr),
        # histogram peak
        (filter_timeshifts.output.filtered_timeshift_maxcorr, hist_peak.input.values),
        (hist_peak.output.histogram_peak, ProcessWorkflow.output.histogram_timeshift_peak),
    ),
    description="find timeshift wf"
)
# find timeshift
refine_regressor_find_timeshift = find_timeshift_wf.copy(description="refine regressor - find timeshift")
# mask 
refine_regressor_mask_timeseries = MaskTimeSeries("refine regressor - mask timeseries")
refine_regressor_mask_timeshifts = MaskTimeSeries("refine regressor - mask timeshift")
# masked peak
masked_hist_peak = HistPeak(description="refine regressor masked histogram peak")
# align depvars
refine_regressor_align_depvars = IteratingNode(AlignTimeSeries(), iterating_inputs=("timeseries", "timeshift"), iterating_name="refine", description="refine regressor - align").setDefaultInputs(refineIter_nr_parallel_processes = -1)
# pca reduce timeseries
refine_regressor_pca_reduce = PCAReducedTimeSeries(description="refine regressor - pca reduce")

# %%
# refine regressor wf
single_refine_regressor_wf = ProcessWorkflow(
    (
        # find timeshift
        (ProcessWorkflow.input._, refine_regressor_find_timeshift.input.all / refine_regressor_find_timeshift.input[("show_pbar", "maxcorr_bipolar", "filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "timeseries_masker")]),
        (ValueNode(None).output.value, refine_regressor_find_timeshift.input[("filter_timeshifts_size", "filter_timeshifts_filter_type", "filter_timeshifts_smooth_fwhm", "timeseries_masker")]),
        (ValueNode(False).output.value, refine_regressor_find_timeshift.input.maxcorr_bipolar),
        (ProcessWorkflow.input.show_pbar_refine, refine_regressor_find_timeshift.input.show_pbar),
        # mask timeseries
        (ProcessWorkflow.input.depvars_timeseries.T, refine_regressor_mask_timeseries.input.timeseries),
        (refine_regressor_find_timeshift.output.depvarsIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeseries.input.mask),
        # mask timeshift
        (refine_regressor_find_timeshift.output.depvarsIter_timeshift_maxcorr, refine_regressor_mask_timeshifts.input.timeseries),
        (refine_regressor_find_timeshift.output.depvarsIter_maxcorr >= ProcessWorkflow.input.refine_regressor_correlation_threshold, refine_regressor_mask_timeshifts.input.mask),
        # masked hist peak
        (refine_regressor_mask_timeshifts.output.masked_timeseries, masked_hist_peak.input.values),
        # align
        (ProcessWorkflow.input.sample_time, refine_regressor_align_depvars.input.time_step),
        (ProcessWorkflow.input.nr_parallel_processes, refine_regressor_align_depvars.input.refineIter_nr_parallel_processes),
        (ProcessWorkflow.input.show_pbar_refine, refine_regressor_align_depvars.input.refineIter_show_pbar),
        (ProcessWorkflow.input.depvars_timeseries.shape[0], refine_regressor_align_depvars.input.length),
        (ValueNode(False).output.value, refine_regressor_align_depvars.input.refineIter_concat_array_outputs),
        (refine_regressor_mask_timeseries.output.masked_timeseries, refine_regressor_align_depvars.input.refineIter_timeseries),
        (masked_hist_peak.output.histogram_peak - refine_regressor_mask_timeshifts.output.masked_timeseries, refine_regressor_align_depvars.input.refineIter_timeshift),
        (ValueNode(False).output.value, refine_regressor_align_depvars.input.fill_nan),
        # pca reduce
        (ProcessWorkflow.input.refine_regressor_explained_variance, refine_regressor_pca_reduce.input.explained_variance),
        (refine_regressor_align_depvars.output.refineIter_aligned_timeseries.T, refine_regressor_pca_reduce.input.timeseries),
        (refine_regressor_pca_reduce.output.reduced_timeseries.mean(axis = 1), ProcessWorkflow.output.refined_regressor_timeseries),
        # refined timeshift is zero since depvars signals have been shifted to timeshift_peak
        (refine_regressor_find_timeshift.output.histogram_timeshift_peak - masked_hist_peak.output.histogram_peak, ProcessWorkflow.output.refined_reference_regressor_timeshift),
    ), description="refine regressor timeseries"
)
# %%
# recursive refine regressor
recursively_refine_regressor = RecursiveNode(single_refine_regressor_wf, recursive_map=(
    (single_refine_regressor_wf.input.regressor_timeseries, single_refine_regressor_wf.output.refined_regressor_timeseries),
    (single_refine_regressor_wf.input.reference_regressor_timeshift, single_refine_regressor_wf.output.refined_reference_regressor_timeshift),
), description="recursively refine regressor timeseries"
)

# %%
##############################################
# dynamic time warping
##############################################
dtw = DTW(description="dynamic time warping")
# reference target correlate
target_reference_correlate = Correlate(description="dtw target reference correlate")
# global regressor align
target_reference_align = AlignTimeSeries(description="dtw target reference regressor")
dynamic_time_warping_wf = ProcessWorkflow(
    (
        # target reference correlate
        (ProcessWorkflow.input.sample_time, target_reference_correlate.input.time_step),
        (ProcessWorkflow.input.target_align_reference_lower_bound, target_reference_correlate.input.lower_limit),
        (ProcessWorkflow.input.target_align_reference_upper_bound, target_reference_correlate.input.upper_limit),
        (ProcessWorkflow.input.correlation_window, target_reference_correlate.input.window),
        (ProcessWorkflow.input.correlation_phat, target_reference_correlate.input.phat),
        (ValueNode(0.0).output.value, target_reference_correlate.input.peak_threshold),
        (ValueNode("max").output.value, target_reference_correlate.input.multi_peak_strategy),
        (ValueNode(None).output.value, target_reference_correlate.input.ref_timeshift),
        (ValueNode(False).output.value, target_reference_correlate.input.bipolar),
        (ProcessWorkflow.input.target_timeseries , target_reference_correlate.input.signal_timeseries_a),
        (ProcessWorkflow.input.reference_timeseries, target_reference_correlate.input.signal_timeseries_b),
        # target reference align
        (ProcessWorkflow.input.sample_time, target_reference_align.input.time_step),
        (ProcessWorkflow.input.target_timeseries.size, target_reference_align.input.length),
        (ProcessWorkflow.input.reference_timeseries, target_reference_align.input.timeseries),
        (target_reference_correlate.output.timeshift_maxcorr, target_reference_align.input.timeshift),
        (ValueNode(False).output.value, target_reference_align.input.fill_nan),
        # dtw
        (ProcessWorkflow.input.sample_time, dtw.input.time_step),
        (ProcessWorkflow.input.dispersion, dtw.input.window),
        (ProcessWorkflow.input.target_timeseries, dtw.input.target_timeseries),
        (target_reference_align.output.aligned_timeseries, dtw.input.reference_timeseries),
        (dtw.output.warped_timeseries, ProcessWorkflow.output.warped_timeseries),
    ), description="dynamic time warping wf"
).setDefaultInputs(dispersion = 10)
# %%
no_dtw = ProcessWorkflow(
    (
        (ProcessWorkflow.input.target_timeseries, ProcessWorkflow.output.warped_timeseries),
    ), description="no dtw"
)
conditional_dtw = ConditionalNode("do_dtw", {True : dynamic_time_warping_wf, False : no_dtw}, False, description="conditional do dtw")

# %% refine regressor wf
refine_regressor_wf = ProcessWorkflow(
    (
        # refine regressor 
        (ProcessWorkflow.input._, recursively_refine_regressor.input["sample_time", "align_regressor_lower_bound", "align_regressor_upper_bound", 'nr_parallel_processes', 'show_pbar', 'correlation_window', 'correlation_phat', 'correlation_peak_threshold', 'correlation_multi_peak_strategy', "refine_regressor_correlation_threshold", 'refine_regressor_explained_variance']),
        (ProcessWorkflow.input.refine_regressor_nr_recursions, recursively_refine_regressor.input.nr_recursions),
        (ProcessWorkflow.input.depvars_timeseries, recursively_refine_regressor.input.depvars_timeseries),
        (ProcessWorkflow.input.regressor_timeseries, recursively_refine_regressor.input.Init_refined_regressor_timeseries),
        (ValueNode(0).output.value, recursively_refine_regressor.input.Init_refined_reference_regressor_timeshift),
        (ValueNode(False).output.value, recursively_refine_regressor.input.show_pbar_refine),
        (recursively_refine_regressor.output.Rec_refined_reference_regressor_timeshift, ProcessWorkflow.output.reference_regressor_timeshift),
        # dynamic time warping regressor to global
        (ProcessWorkflow.input._, conditional_dtw.input[('sample_time', 'correlation_phat', 'correlation_window', 'do_dtw')]),
        (ProcessWorkflow.input.align_regressor_lower_bound, conditional_dtw.input.target_align_reference_lower_bound),
        (ProcessWorkflow.input.align_regressor_upper_bound, conditional_dtw.input.target_align_reference_upper_bound),
        (ProcessWorkflow.input.dtw_dispersion, conditional_dtw.input.dispersion),
        (ProcessWorkflow.input.regressor_timeseries, conditional_dtw.input.reference_timeseries),
        (recursively_refine_regressor.output.Rec_refined_regressor_timeseries, conditional_dtw.input.target_timeseries),
        (conditional_dtw.output.warped_timeseries, ProcessWorkflow.output.refined_regressor_timeseries),        
    ),
    description="refine regressor wf"
).setDefaultInputs(refine_regressor_correlation_threshold = 0.75, refine_regressor_nr_recursions = 0, refine_regressor_explained_variance = 0.8)
#%%