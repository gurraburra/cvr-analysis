from process_control import ProcessNode
from nilearn import image, maskers
import pandas as pd
import re
from glob import glob
import os
import json

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
    
def _getBIDSFiles(bids_dir, subject, session=None, task=None, run=None, space=None, desc=None, suffix=None, extension=None):
    # base dir
    if session is None:
        base_dir = os.path.join(bids_dir, f"sub-{subject}", "*")
    else:
        base_dir = os.path.join(bids_dir, f"sub-{subject}", f"ses-{session}", "*")
    # file specification
    file_spec = f"sub-{subject}"
    # add optional specifyers
    if session is not None:
        file_spec += f"_ses-{session}"
    if task is not None:
        file_spec += f"_task-{task}"
    if run is not None:
        file_spec += f"_run-{run}"
    if space is not None:
        file_spec += f"_space-{space}"
    # add wild card
    file_spec += "*"
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

    return glob(os.path.join(base_dir, file_spec))
    

class LoadBOLDData(ProcessNode):
    outputs = ("bold_img", "confounds_df", "events_df", "tr", "nr_measurements")
    
    def _run(self, bids_dir : str, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, load_confounds : bool = False, load_events : bool = True) -> dict:
        # load bild data
        bold_img = image.load_img(
                        _checkBIDSQuery("BOLD", 
                            _getBIDSFiles(
                                        bids_dir,
                                            subject=subject, 
                                                session=session, 
                                                    task=task, 
                                                        run=run, 
                                                            space=space, 
                                                                suffix="bold", 
                                                                    extension=".nii.gz")))
        # nr_meas
        nr_meas = bold_img.shape[3]

        bold_json_file = _checkBIDSQuery("BOLD JSON", 
                            _getBIDSFiles(
                                        bids_dir,
                                            subject=subject, 
                                                session=session, 
                                                    task=task, 
                                                        run=run, 
                                                            space=space, 
                                                                suffix="bold", 
                                                                    extension=".json"))
    
        # tr
        # tr = bids_dir.get_metadata(bold_img.get_filename())["RepetitionTime"]
        with open(bold_json_file, "r") as file:
            tr = json.load(file)["RepetitionTime"]
        
        # load in confounds 
        if load_confounds:
            confounds_df = pd.read_csv(
                                _checkBIDSQuery("confounds", 
                                    _getBIDSFiles(
                                        bids_dir,
                                            subject=subject, 
                                                session=session, 
                                                    task=task, 
                                                        run=run, 
                                                            suffix="timeseries", 
                                                                extension=".tsv")), sep='\t')
        else:
            confounds_df = None
        
        # load in events
        if load_events:
            events_df = pd.read_csv(
                                _checkBIDSQuery("events", 
                                     _getBIDSFiles(
                                        bids_dir, 
                                            subject=subject, 
                                                session=session, 
                                                    task=task, 
                                                        run=run, 
                                                            suffix="events", 
                                                                extension=".tsv")), sep='\t')
        else:
            events_df = None

        
        
        return bold_img, confounds_df, events_df, tr, nr_meas


class LoadBidsImg(ProcessNode):
    outputs = ("bids_img", )
    
    def _run(self, bids_dir : str, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, desc : str = None, suffix : str = None) -> dict:
        # check if is nifti file
        if _isNiftiFile(desc):
            bids_file = desc
        else:
            # if not, assume it is a description
            bids_file = _checkBIDSQuery(suffix, 
                         _getBIDSFiles(
                            bids_dir, 
                                subject=subject, 
                                    session=session, 
                                        task=task, 
                                            run=run, 
                                                space=space, 
                                                    desc=desc,
                                                        suffix=suffix, 
                                                            extension=".nii.gz"))
        # load in bids data
        bids_img = image.load_img(bids_file)
        
        return bids_img, 


class CropBOLDImg(ProcessNode):
    outputs = ("cropped_bold_img",)

    def _run(self, bold_img, voxel_mask_img):
        # make sure they are aligned
        tmp_mask = image.resample_to_img(voxel_mask_img, bold_img, interpolation='nearest')
        return image.math_img('img * np.array(voxel_mask, dtype=bool)[...,None]', img = bold_img, voxel_mask = tmp_mask), 

class VoxelTimeSeriesMasker(ProcessNode):
    outputs = ("voxel_masker",)
    
    def _run(self, voxel_mask_img, spatial_smoothing_fwhm = None) -> tuple:
        return maskers.NiftiMasker(mask_img=voxel_mask_img, smoothing_fwhm=spatial_smoothing_fwhm)
    
class RoiTimeSeriesMasker(ProcessNode):
    outputs = ("roi_masker",)
    
    def _run(self, labels_img, voxel_mask_img, spatial_smoothing_fwhm = None) -> tuple:
        return maskers.NiftiLabelsMasker(labels_img=labels_img, mask_img=voxel_mask_img, smoothing_fwhm=spatial_smoothing_fwhm)

class GetTimeSeriesEvent(ProcessNode):
    outputs = ("times", "timeseries", "event_name")
    
    def _run(self, events_df : pd.DataFrame, event_name) -> tuple:
        # inner search function
        def search(name_):
            for event_type in set(events_df["trial_type"]):
                if re.search(name_, event_type):
                    times, ts = events_df.query(f"trial_type == '{event_type}'").loc[:,["onset", "value"]].to_numpy().T
                    return times, ts, event_type
                
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