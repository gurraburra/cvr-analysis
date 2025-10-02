from process_control import ProcessNode
from nilearn import image, maskers
import pandas as pd
import re
from glob import glob
import os
import json
import numpy as np

def _checkBIDSQuery(file : str, files : list[str]) -> str:
    if len(files) == 0:
        raise ValueError(f"Could not find {file} file.")
    elif len(files) > 1:
        raise ValueError(f"Found multiple {file} files.")
    else:
        return files[0]
    
def _isNiftiFile(desc):
    if desc is not None:
        return re.search(".nii(.gz)?$", desc) is not None
    else:
        return desc
    
def _getBIDSFiles(bids_directory, subject, session=None, data_type=None, task=None, acq=None, run=None, space=None, recording=None, desc=None, suffix=None, extension=None):
    # data type
    if data_type is None:
        data_folder = "*"
    else:
        data_folder = data_type
    # base dir
    if session is None:
        base_dir = os.path.join(bids_directory, f"sub-{subject}", data_folder)
    else:
        base_dir = os.path.join(bids_directory, f"sub-{subject}", f"ses-{session}", data_folder)
    # file specification
    file_spec = f"sub-{subject}"
    # add optional specifyers
    if session is not None:
        file_spec += f"_ses-{session}"
    if task is not None:
        file_spec += f"_task-{task}"
    if acq is not None:
        file_spec += f"_acq-{acq}"
    if run is not None:
        file_spec += f"_run-{run}"
    if space is not None:
        file_spec += f"_space-{space}"
    # add wild card
    file_spec += "*"
    # recording
    if recording is not None:
        file_spec += f"_recording-{recording}"
    # desc
    if desc is not None:
        file_spec += f"_desc-{desc}"
    # suffix
    if suffix is not None:
        file_spec += f"_{suffix}"
    elif file_spec[-1] != '*':
        file_spec += '*'
    # extension
    if extension is not None:
        file_spec += f"{extension}"
    elif file_spec[-1] != '*':
        file_spec += '*'
    # seach pattern
    search_pattern = os.path.join(base_dir, file_spec)
    return glob(search_pattern)
    

class LoadBOLDData(ProcessNode):
    outputs = ("bold_img", "confounds_df", "tr", "nr_measurements")
    
    def _run(self, bids_directory : str, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, load_confounds : bool = False) -> dict:
        # load bild data
        bold_img = image.load_img(
                        _checkBIDSQuery("BOLD", 
                            _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type="func",
                                                        task=task, 
                                                            run=run, 
                                                                space=space, 
                                                                    suffix="bold", 
                                                                        extension=".nii*")))
        # nr_meas
        nr_meas = bold_img.shape[3]

        bold_physio_json = _checkBIDSQuery("BOLD JSON", 
                            _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type="func",
                                                        task=task, 
                                                            run=run, 
                                                                space=space, 
                                                                    suffix="bold", 
                                                                        extension=".json"))
    
        # tr
        # tr = bids_directory.get_metadata(bold_img.get_filename())["RepetitionTime"]
        with open(bold_physio_json, "r") as file:
            tr = json.load(file)["RepetitionTime"]
        
        # load in confounds 
        if load_confounds:
            confounds_df = pd.read_csv(
                                _checkBIDSQuery("confounds", 
                                    _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type="func",
                                                        task=task, 
                                                            run=run, 
                                                                suffix="timeseries", 
                                                                    extension=".tsv")), sep='\t')
        else:
            confounds_df = None
        
        

        
        
        return bold_img, confounds_df, tr, nr_meas


class LoadBidsImg(ProcessNode):
    outputs = ("bids_img", )
    
    def _run(self, bids_directory : str, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, desc : str = None, suffix : str = None) -> dict:
        # check if is nifti file
        if _isNiftiFile(desc):
            bids_file = desc
        else:
            # if not, assume it is a description
            bids_file = _checkBIDSQuery(suffix, 
                         _getBIDSFiles(
                            bids_directory, 
                                subject=subject, 
                                    session=session, 
                                        task=task, 
                                            run=run, 
                                                space=space, 
                                                    desc=desc,
                                                        suffix=suffix, 
                                                            extension=".nii*"))
        # load in bids data
        bids_img = image.load_img(bids_file)
        
        return bids_img, 


class CropBOLDImg(ProcessNode):
    outputs = ("cropped_bold_img", "resampled_voxel_mask_img")

    def _run(self, bold_img, voxel_mask_img):
        # make sure they are aligned
        resampled_voxel_mask_img = image.resample_to_img(voxel_mask_img, bold_img, interpolation='nearest', force_resample=True, copy_header=True)
        return image.math_img('img * np.array(voxel_mask, dtype=bool)[...,None]', img = bold_img, voxel_mask = resampled_voxel_mask_img), resampled_voxel_mask_img

class VoxelTimeSeriesMasker(ProcessNode):
    outputs = ("voxel_masker",)
    
    def _run(self, voxel_mask_img, spatial_smoothing_fwhm = None) -> tuple:
        return maskers.NiftiMasker(mask_img=voxel_mask_img, smoothing_fwhm=spatial_smoothing_fwhm)
    
class RoiTimeSeriesMasker(ProcessNode):
    outputs = ("roi_masker",)
    
    def _run(self, labels_img, voxel_mask_img, spatial_smoothing_fwhm = None) -> tuple:
        return maskers.NiftiLabelsMasker(labels_img=labels_img, mask_img=voxel_mask_img, smoothing_fwhm=spatial_smoothing_fwhm, keep_masked_labels = False)

class LoadTimeseriesEvent(ProcessNode):
    # keept for backwards compatibility with old events file
    outputs = ("times", "timeseries", "event_name", "unit")
    
    def _run(self, bids_directory : str, subject : str = None, session : str = None, data_type : str = None, task : str = None, run : str = None, event_name : list = None) -> tuple:
        # load in events
        for acq in ["*", None]:
            # try events using acq
            try:
                events_df = pd.read_csv(
                                    _checkBIDSQuery("events", 
                                        _getBIDSFiles(
                                            bids_directory, 
                                                subject=subject, 
                                                    session=session, 
                                                        data_type=data_type,
                                                            task=task, 
                                                                acq=acq,
                                                                    run=run, 
                                                                        suffix="events", 
                                                                            extension=".tsv")), sep='\t')
            except ValueError:
                continue
            break
        else:
            raise ValueError("Could not load events file.")
        
        # inner search function
        def search(name_):
            for event_type in set(events_df["trial_type"]):
                if re.search(name_, event_type):
                    times, ts = events_df.query(f"trial_type == '{event_type}'").loc[:,["onset", "value"]].to_numpy().T
                    # try to get unit
                    u = re.search(r".*\[(.*)\]$", event_type)
                    if u:
                        unit = u.group(1)
                    else:
                        unit = ""
                    return times, ts, event_type, unit
                
        # check if list or tuple given
        if isinstance(event_name, (list, tuple)):
            for event_n in event_name:
                r = search(event_n)
                if r is not None:
                    return r
        else:
            r = search(event_name)
            if r is not None:
                return r
        raise ValueError(f"Could not find time series event '{event_name}'.")
    

class LoadDopplerData(ProcessNode):
    outputs = ("times", "mean_tr", "time_unit", "blood_flow_ts", "blood_flow_headers", "blood_flow_units", "events_df")
    
    def _run(self, bids_directory : str, subject : str = None, session : str = None, task : str = None, run : str = None, add_global_signal : bool = True) -> dict:
        # load doppler data
        doppler_ts = pd.read_csv(
                        _checkBIDSQuery("Doppler", 
                            _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type="doppler",
                                                        task=task, 
                                                            run=run, 
                                                                suffix="doppler", 
                                                                    extension=".tsv")),
                                                                        sep='\t', header=0, index_col=False)
        # get headers and units
        headers = []
        units = []
        for c in doppler_ts.columns:
            for p in [r"(.*) \[(.*)\]", r"(.*) \((.*)\)"]:
                g = re.match(p, c)
                if g is not None:
                    headers.append(g.group(1))
                    units.append(g.group(2))
                    break
            # check if no match happened
            if g is None:
                headers.append(c)
                units.append("")
        # check time in headers
        assert "time" in headers, "Missing mandatory column 'time' in doppler data file"
        # get indices for time a blood flow timeseries
        time_idx = headers.index("time")
        bf_idxs = list(range(len(headers)))
        del bf_idxs[time_idx]
        assert len(bf_idxs) != 0, "No doppler signals in data"
        # convert to numpy
        doppler_ts = doppler_ts.to_numpy() 
        headers = np.array(headers)
        units = np.array(units)
        # get timeseries, headers and units
        doppler_time_ts, header_time, unit_time = doppler_ts[:,time_idx], headers[time_idx], units[time_idx]
        doppler_bf_ts, headers_bf, units_bf = doppler_ts[:,bf_idxs], headers[bf_idxs], units[bf_idxs]
        # add global bf ts
        if add_global_signal:
            assert np.unique(units_bf) != 1, "Can't add global doppler signal, different units."
            doppler_bf_ts = np.hstack((doppler_bf_ts, doppler_bf_ts.mean(axis = 1)[:,None]))
            headers_bf = np.append(headers_bf, "global")
            units_bf = np.append(units_bf, units_bf[0])
        
        # mean tr
        mean_tr = np.diff(doppler_time_ts).mean()

        # load in events
        for acq in ["doppler", None]:
            # try events using acq
            try:
                events_df = pd.read_csv(
                                    _checkBIDSQuery("events", 
                                        _getBIDSFiles(
                                            bids_directory, 
                                                subject=subject, 
                                                    session=session, 
                                                        data_type="doppler",
                                                            task=task, 
                                                                acq=acq,
                                                                    run=run, 
                                                                        suffix="events", 
                                                                            extension=".tsv")), sep='\t')
            except ValueError:
                continue
            break
        else:
            raise ValueError("Could not load events file.")

        
        
        return doppler_time_ts, mean_tr, unit_time, doppler_bf_ts, headers_bf, units_bf, events_df 
    

class LoadPhysioData(ProcessNode):
    outputs = ("times", "timeseries", "time_step", "units", "variables")
    def _run(self, bids_directory : str, subject : str = None, session : str = None, data_type : str = None, task : str = None, acq : str = None, run : str = None, recording : str = None, variables : list = None) -> dict:
        # load json
        physio_json_name =  _checkBIDSQuery("Physio", 
                            _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type=data_type,
                                                        task=task, 
                                                            acq=acq,
                                                                run=run, 
                                                                    recording=recording,
                                                                        suffix="physio", 
                                                                            extension=".json"))
        with open(physio_json_name, "r") as j_file:
            physio_json = json.load(j_file)
        # check file
        assert "SamplingFrequency" in physio_json; samp_freq = physio_json["SamplingFrequency"]
        assert "StartTime" in physio_json; start_time = physio_json["StartTime"]
        assert "Columns" in physio_json; columns = physio_json["Columns"]

        # load in physio ts
        physio_ts = np.loadtxt(
                        _checkBIDSQuery("Physio", 
                            _getBIDSFiles(
                                        bids_directory,
                                            subject=subject, 
                                                session=session, 
                                                    data_type=data_type,
                                                        task=task, 
                                                            acq=acq,
                                                                run=run, 
                                                                    recording=recording,
                                                                        suffix="physio", 
                                                                            extension=".tsv*")), 
                                                                                dtype=float, delimiter='\t')
        
        # check variables
        if variables is None:
            variables = tuple(columns)
            single_var = False
        elif not isinstance(variables, (list,tuple)):
            variables = (variables,)
            single_var = True
        else:
            single_var = False
        # mask data
        mask = np.zeros(len(columns), dtype=bool)
        units = []
        for var in variables:
            try:
                mask[columns.index(var)] = True
                units.append(physio_json[var]["Units"])
            except ValueError as exc:
                raise ValueError(f"missing variable '{var}' in physio file '{recording}'") from exc
            
        # times
        t = np.arange(0,physio_ts.shape[0]) * 1 / samp_freq + start_time

        # single var?
        if single_var:
            return t, physio_ts[:,mask][:,0], 1 / samp_freq, units[0], variables[0]
        else:
            return t, physio_ts[:,mask], 1 / samp_freq, units, variables