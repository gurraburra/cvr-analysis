from threadpoolctl import threadpool_limits
import argparse
from tqdm.auto import tqdm
from tqdm.contrib.itertools import product as tqdm_product
from itertools import product as iter_product
from multiprocess import cpu_count
from functools import partial
from scipy.stats.qmc import Sobol
import numpy as np
import warnings
import pandas as pd
import ast
import pprint

def handleNone(type_, arg):
    if arg == "None":
        return None
    else:
       return type_(arg)
    
def handleBool(arg):
    if arg == "True":
        return True
    elif arg == "False":
        return False
    else:
        raise ValueError(f"could not convert string to bool: '{arg}'")
    
def handleTuple(type_, arg, ensure_tuple = True, ensure_len_2 = True):
    arg = ast.literal_eval(arg)
    if arg is not None:
        if isinstance(arg, (tuple,list)):
            if ensure_len_2:
                if len(arg) != 2:
                    raise ValueError(f"expected tuple with length 2 but a tuple of length {len(arg)} was given: '{arg}'")
            return tuple(type_(a) if a is not None else a for a in arg)
        else:
            if ensure_tuple:
                raise ValueError(f"expected tuple but a {type(arg)} was given: '{arg}'")
            return type_(arg)
    else:
        return arg

def main():
    # parse argument
    parser = argparse.ArgumentParser(description='Seed analysis')
    parser.add_argument('bids_dir', type=str, help='bids folder with data')
    parser.add_argument('output_dir', type=str, help='output folder')
    parser.add_argument('analysis_level', type=str, choices=["participant"], default="participant", nargs="?", help='which level of analysis')
    parser.add_argument('--nprocs', type=int, default=-1, help='number of cpus to be used')
    parser.add_argument('--omp-threads', '--nthreads-per-proc', type=partial(handleNone, int), help='number of threads per process')
    parser.add_argument('--force-run', action='store_true', help='flag specifing if old data files should be overriden')
    parser.add_argument('--full-output', action='store_true', help='flag specifing if all output files should be saved')
    parser.add_argument('--verbose', action='store_true', help='verbose printing or not')
    
    # subject
    parser.add_argument('--participant-label', type=str, action="extend", nargs="+", help='participant to analyze')
    # session
    parser.add_argument('--session-label', type=partial(handleNone, str), action="extend", nargs="+", help='session to analyze')
    # task
    parser.add_argument('--task', type=partial(handleNone, str), action="extend", nargs="+", help='which task')
    # run
    parser.add_argument('--run', type=partial(handleNone, str), action="extend", nargs="+", help='which run')
    # space
    parser.add_argument('--space', type=partial(handleNone, str), action="extend", nargs="+", help='which space for analysis')
    # include confounds
    parser.add_argument('--include-confounds', type=handleBool, action="extend", nargs="+", help='include confounding variables in analysis')
    # voxel mask
    parser.add_argument('--voxel-mask', type=partial(handleNone, str), action="extend", nargs="+", help='mask for bold data')
    # roi mask
    parser.add_argument('--roi-masker', type=partial(handleNone, str), action="extend", nargs="+", help='roi mask for timeseries')
    # smoothing
    parser.add_argument('--spatial-smoothing-fwhm', type=partial(handleNone, float), action="extend", nargs="+", help='spatial smoothing full-width half-maximum in mm')
    # min sampling freq
    parser.add_argument('--min-sample-freq', type=partial(handleNone, float), action="extend", nargs="+", help='minimum sample frequency')
    # analysis bounds
    parser.add_argument('--analysis-bounds', type=partial(handleTuple, float), action="extend", nargs="+", help='pair of lower and upper bounds for analysis (in seconds)')
    # detrend linear order
    parser.add_argument('--detrend-linear-order', type=partial(handleNone, int), action="extend", nargs="+", help='detrend linear order')
    # filter freq
    parser.add_argument('--temporal-filter-freq', type=partial(handleTuple, float, ensure_tuple = False), action="extend", nargs="+", help='single filter value give lowpass filter and tuple give band-filter')
    # baseline strategy
    parser.add_argument('--baseline-strategy', type=str, action="extend", nargs="+", help='strategy to calculate baseline')
    # use co2 regressor
    parser.add_argument('--use-co2-regressor', type=handleBool, action="extend", nargs="+", help='use co2 regressor')
    # initial global align bounds
    parser.add_argument('--global-align-co2-bounds', type=partial(handleTuple, float), action="extend", nargs="+", help='pair of lower and upper bounds for initial aligning regressor to global signal (in seconds)')
    # align regressor bounds
    parser.add_argument('--align-regressor-bounds', type=partial(handleTuple, float), action="extend", nargs="+", help='pair of lower and upper bounds for aligning regressor (in seconds)')
    # bipolar correlation
    parser.add_argument('--maxcorr-bipolar', type=handleBool, action="extend", nargs="+", help='bipolar correlation')
    # correlation window
    parser.add_argument('--correlation-window', type=partial(handleNone, str), action="extend", nargs="+", help='correlation window')
    # correlation multi peak
    parser.add_argument('--correlation-multi-peak-strategy', type=partial(handleNone, str), action="extend", nargs="+", help='strategy to decide between multiple peaks in the cross-correlation function, either "ref", "max" or "mi"')
    # correlation peak threshold
    parser.add_argument('--correlation-peak-threshold', type=partial(handleNone, float), action="extend", nargs="+", help='threshold for peak detection in correlation function')
    # filter timeshifts type
    parser.add_argument('--filter-timeshifts-filter-type', type=partial(handleNone, str), action="extend", nargs="+", help='type of filter, either "mean", "median" or "None"')
    # filter timeshifts size
    parser.add_argument('--filter-timeshifts-size', type=partial(handleNone, int), action="extend", nargs="+", help='size of filter window (in voxels) used in filtering of timeshifts')
    # filter timeshifts smoothing fwhm
    parser.add_argument('--filter-timeshifts-smooth-fwhm', type=partial(handleNone, float), action="extend", nargs="+", help='fwhm smoothing of timeshifts')
    # refine regressor nr recursions
    parser.add_argument('--refine-regressor-nr-recursion', type=partial(handleNone, int), action="extend", nargs="+", help='refine regressor nr recursions')
    # refine regressor threshold
    parser.add_argument('--refine-regressor-correlation-threshold', type=partial(handleNone, float), action="extend", nargs="+", help='lower correlation threshold for voxels to be used in refining regressor')
    # refine regressor explained variance
    parser.add_argument('--refine-regressor-explained-variance', type=partial(handleNone, float), action="extend", nargs="+", help='how much variance to be explained by pca components when refining regressor')
    # ensure co2 units
    parser.add_argument('--ensure-co2-units', type=handleBool, action="extend", nargs="+", help='ensure cvr maps are in co2 units by applying DTW if necessary')
    # confound regressor correaltion threshold
    parser.add_argument('--confound-regressor-correlation-threshold', type=partial(handleNone, float), action="extend", nargs="+", help='confound regressor correlation threshold')

    # sampling options
    # sobol sampling
    parser.add_argument('--sobol-sampling', type=str, action="extend", nargs="+", help='give number of sobol samples plus name of factors to be included')
    # parameter list file
    parser.add_argument('--parameter-list-file', type=str, help='file containing parameter list')


    args = parser.parse_args()

    # check particpant label
    if args.participant_label is None:
        raise ValueError("No participant label given.")
    else:
        sub_options = args.participant_label
    # check session label
    if args.session_label is None:
        ses_options = [None]
    else:
        ses_options = args.session_label
    # task
    if args.task is None:
        task_options = [None]
    else:
        task_options = args.task
    # run
    if args.run is None:
        run_options = [None]
    else:
        run_options = args.run
    # space
    if args.space is None:
        space_options = [None]
    else:
        space_options = args.space
    # include confounds
    if args.include_confounds is None:
        include_confounds_options = [True]
    else:
        include_confounds_options = args.include_confounds
    # voxel mask
    if args.voxel_mask is None:
        voxel_mask_options = ["brain"]
    else:
        voxel_mask_options = args.voxel_mask
    # roi mask
    if args.roi_masker is None:
        roi_masker_options = [None]
    else:
        roi_masker_options = args.roi_masker
    # analysis bounds
    if args.analysis_bounds is None:
        analysis_bounds_options = [(None, None)]
    else:
        analysis_bounds_options = args.analysis_bounds
    # spatial_smoothing
    if args.spatial_smoothing_fwhm is None:
        smoothing_fwhm_options = [5.0]
    else:
        smoothing_fwhm_options = args.spatial_smoothing_fwhm
    # min-sample-freq
    if args.min_sample_freq is None:
        min_sample_freq_options = [2]
    else:
        min_sample_freq_options = args.min_sample_freq
    # detrend
    if args.detrend_linear_order is None:
        detrend_linear_order_options = [None]
    else:
        detrend_linear_order_options = args.detrend_linear_order
    # temporal filter
    if args.temporal_filter_freq is None:
        temporal_filter_freq_options = [None]
    else:
        temporal_filter_freq_options = args.temporal_filter_freq
    # baseline strategy
    if args.baseline_strategy is None:
        baseline_strategy_options = ['overall-mean']
    else:
        baseline_strategy_options = args.baseline_strategy
    # co2_options
    if args.use_co2_regressor is None:
        co2_options = [True]
    else:
        co2_options = args.use_co2_regressor
    # global align co2 bounds
    if args.global_align_co2_bounds is None:
        global_align_co2_bounds_options = [(None, None)]
    else:
        global_align_co2_bounds_options = args.global_align_co2_bounds
    # align regressor bounds
    if args.align_regressor_bounds is None:
        align_regressor_bounds_options = [(None, None)]
    else:
        align_regressor_bounds_options = args.align_regressor_bounds
    # bipolar options
    if args.maxcorr_bipolar is None:
        bipolar_options = [True]
    else:
        bipolar_options = args.maxcorr_bipolar
    # correlation options
    if args.correlation_window is None:
        correlation_window_options = [None]
    else:
        correlation_window_options = args.correlation_window
    # correlation multi peak
    if args.correlation_multi_peak_strategy is None:
        correlation_multi_peak_strategy_options = [None]
    else:
        correlation_multi_peak_strategy_options = args.correlation_multi_peak_strategy
    # correlation multi peak
    if args.correlation_peak_threshold is None:
        correlation_peak_threshold_options = [0.0]
    else:
        correlation_peak_threshold_options = args.correlation_peak_threshold
    # filter timeshifts type
    if args.filter_timeshifts_filter_type is None:
        filter_timeshifts_filter_type_options = [None]
    else:
        filter_timeshifts_filter_type_options = args.filter_timeshifts_filter_type
    # filter timeshifts size
    if args.filter_timeshifts_size is None:
        filter_timeshifts_size_options = [3]
    else:
        filter_timeshifts_size_options = args.filter_timeshifts_size
    # filter timeshifts smooth fwhm
    if args.filter_timeshifts_smooth_fwhm is None:
        filter_timeshifts_smooth_fwhm_options = [None]
    else:
        filter_timeshifts_smooth_fwhm_options = args.filter_timeshifts_smooth_fwhm
    # refine regressor nr recursions
    if args.refine_regressor_nr_recursion is None:
        refine_regressor_nr_recursion_options = [0]
    else:
        refine_regressor_nr_recursion_options = args.refine_regressor_nr_recursion
    # refine regressor threshold
    if args.refine_regressor_correlation_threshold is None:
        refine_regressor_correlation_threshold_options = [0.5]
    else:
        refine_regressor_correlation_threshold_options = args.refine_regressor_correlation_threshold
    # refine regressor explained variance
    if args.refine_regressor_explained_variance is None:
        refine_regressor_explained_variance_options = [0.8]
    else:
        refine_regressor_explained_variance_options = args.refine_regressor_explained_variance
    # ensure co2 units
    if args.ensure_co2_units is None:
        ensure_co2_units_options = [False]
    else:
        ensure_co2_units_options = args.ensure_co2_units
    # confound correlation threshold
    if args.confound_regressor_correlation_threshold is None:
        confound_regressor_correlation_thr_options = [None]
    else:
        confound_regressor_correlation_thr_options = args.confound_regressor_correlation_threshold
    
    # get number of cores to use
    if args.nprocs <= -1:
         nprocs = cpu_count()
    else:
         nprocs = args.nprocs
    
    # option names
    options = {
        "subject" : sub_options,
        "session" : ses_options,
        "task" : task_options,
        "run" : run_options,
        "space" : space_options,
        "include-confounds" : include_confounds_options,
        "voxel-mask" : voxel_mask_options,
        "roi-masker" : roi_masker_options,
        "spatial-smoothing-fwhm" : smoothing_fwhm_options,
        "min-sample-freq" : min_sample_freq_options,
        "analysis-bounds" : analysis_bounds_options,
        "detrend-linear-order" : detrend_linear_order_options,
        "temporal-filter-freq" : temporal_filter_freq_options,
        "baseline-strategy" : baseline_strategy_options,
        "use-co2-regressor" : co2_options,
        "global-align-co2-bounds" : global_align_co2_bounds_options,
        "align-regressor-bounds" : align_regressor_bounds_options,
        "maxcorr-bipolar" : bipolar_options,
        "correlation-window" : correlation_window_options,
        "correlation-multi-peak-strategy"           : correlation_multi_peak_strategy_options,
        "correlation-peak-threshold"                : correlation_peak_threshold_options,
        "ensure-co2-units"                          : ensure_co2_units_options,
        "refine-regressor-nr-recursions"            : refine_regressor_nr_recursion_options,
        "refine-regressor-correlation-threshold"    : refine_regressor_correlation_threshold_options,
        "refine-regressor-explained-variance"       : refine_regressor_explained_variance_options,
        "filter-timeshifts-filter-type"             : filter_timeshifts_filter_type_options,
        "filter-timeshifts-size"                    : filter_timeshifts_size_options,
        "filter-timeshifts-smooth-fwhm"             : filter_timeshifts_smooth_fwhm_options,
        "confound-regressor-correlation-threshold"  : confound_regressor_correlation_thr_options,
    }

    ordered_factors = tuple(options.keys())

    # check additional parameter options
    if args.parameter_list_file is not None:
        # read in parameter list
        parameter_list = pd.read_csv(args.parameter_list_file, sep='\t')
        print(parameter_list.shape)
        # check all parameters are in options
        for c in parameter_list.columns:
            assert c in options, f"Parameter '{c}' is not a valid option: {' ,'.join(ordered_factors)}"
        # check before conversion
        def applyLiteralEvalStr(val):
            if isinstance(val, str):
                return ast.literal_eval(val)
            else:
                return val
        # handle conversion
        converters={"global-align-co2-bounds": applyLiteralEvalStr, "align-regressor-bounds": applyLiteralEvalStr, "analysis-bounds": applyLiteralEvalStr, "temporal-filter-freq": applyLiteralEvalStr}
        for c_name, func in converters.items():
            if c_name in parameter_list.columns:
                parameter_list[c_name] = parameter_list[c_name].apply(func)
        # replace nan with None
        parameter_list = parameter_list.fillna(np.nan).replace([np.nan], [None])
        print(parameter_list.shape)
        # product options
        product_options = {f : options[f] for f in options if f not in parameter_list.columns}
        print(product_options)
        product_samples = iter_product(*product_options.values())
        print(len(tuple(iter_product(*product_options.values()))))
        # create iterations
        iters = np.array([prod + par_list for prod,par_list in iter_product(product_samples, tuple(parameter_list.itertuples(index=False, name=None)))], dtype=object)
        print(iters.shape)
        # reorder factors
        factor_names = tuple(product_options.keys()) + tuple(parameter_list.columns)
        iters = iters[:,[factor_names.index(factor) for factor in ordered_factors]]
    elif args.sobol_sampling is not None:
        # check 
        sobol_parser = argparse.ArgumentParser(description="Parse sobol factors")
        sobol_parser.add_argument('nr_samples', type=int, help='number of samples')
        for opt in options:
            sobol_parser.add_argument(f"--{opt}", action="store_true")
        # parse sobol arguments
        sobol_args = sobol_parser.parse_args(args.sobol_sampling[:1] +  [f"--{f}" for f in args.sobol_sampling[1:]])
        # sobol options
        sobol_options = {f : options[f] for f in options if sobol_args.__getattribute__(f.replace("-","_"))}
        # sobol factors
        nr_levels = np.array([len(opt) for _, opt in sobol_options.items()])
        max_sobol_iters = np.prod(nr_levels)
        # ignore warnings from Sobol sampler (not using power of 2 number of samples for balance sampling)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # create sampler
            sampler = Sobol(d=len(sobol_options), scramble=False)
            # create samples
            sobol_idx = np.unique(np.floor(sampler.random(sobol_args.nr_samples) * nr_levels).astype(int), axis = 0)
            # make sure the numer of samples requested are computed
            while sobol_idx.shape[0] < sobol_args.nr_samples and sobol_idx.shape[0] < max_sobol_iters:
                sobol_idx = np.concatenate((sobol_idx, np.unique(np.floor(sampler.random(sobol_args.nr_samples - sobol_idx.shape[0]) * nr_levels).astype(int), axis = 0)), axis = 0)
        # create sobol samples
        sobol_options_list = tuple(sobol_options.values())
        sobol_samples = tuple(tuple(sobol_options_list[col][s_idx] for col, s_idx in enumerate(sobol_idx[row])) for row in range(sobol_idx.shape[0]))
        # product options
        product_options = {f : options[f] for f in options if not sobol_args.__getattribute__(f.replace("-","_"))}
        product_samples = iter_product(*product_options.values())
        # create iterations
        iters = np.array([prod + sob for prod,sob in iter_product(product_samples, sobol_samples)], dtype=object)
        # reorder factors
        factor_names = tuple(product_options.keys()) + tuple(sobol_options.keys())
        iters = iters[:,[factor_names.index(factor) for factor in ordered_factors]]
    else:
        # use itertools product to run all combination of options
        iters = np.array(tuple(iter_product(*options.values())), dtype=object)
    # sort factors
    iters = iters[np.lexsort(np.vectorize(str)(iters[:,::-1].T))]

    # load in from separate module
    from cvr_analysis.modalities_study.setup_wf import cvr_wf

    # print
    print("----- CVR analysis -----")
    print()
    print(f"Using {nprocs} processes and omp-threads limited to {args.omp_threads}.")
    print("--------------------")
    print()
    print(f"Number of iterations: {len(iters)}")
    print("--------------------")

    # limit nr threads using threadpoolctl
    with threadpool_limits(limits=args.omp_threads):
        # loop through iterations
        for iter_ in tqdm(iters, miniters = 1):
            # replace - with _
            iter_args_dict = {factor.replace("-","_") : value for factor, value in zip(ordered_factors,iter_)}

            # unpack analysis bounds
            iter_args_dict["analysis_start_time"], iter_args_dict["analysis_end_time"] = iter_args_dict["analysis_bounds"]
            del iter_args_dict["analysis_bounds"]
            # unpack align regressor bounds
            iter_args_dict["align_regressor_lower_bound"], iter_args_dict["align_regressor_upper_bound"] = iter_args_dict["align_regressor_bounds"]
            del iter_args_dict["align_regressor_bounds"]
            # unpack initial global align bounds
            iter_args_dict["global_align_co2_lower_bound"], iter_args_dict["global_align_co2_upper_bound"] = iter_args_dict["global_align_co2_bounds"]
            del iter_args_dict["global_align_co2_bounds"]

            # print args if verbose
            if args.verbose:
                print()
                print("Iterating arguments:")
                pprint.pp(iter_args_dict, sort_dicts=True)
                print()

            # run workflow
            cvr_wf.run(ignore_cache = False, save_data = True, nr_parallel_processes = nprocs,
                                bids_dir = args.bids_dir, verbose = args.verbose, force_run = args.force_run, full_output = args.full_output, output_directory = args.output_dir, 
                                        **iter_args_dict)

    # done
    if args.verbose:
        print()
        print("--- Done ---")
        print()

# main script
if __name__ == "__main__":
    main()