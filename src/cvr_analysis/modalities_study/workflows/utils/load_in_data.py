from process_control import ProcessNode
from bids import BIDSLayout
from nilearn import image, masking, maskers
import pandas as pd
import nibabel
import numpy as np
import re

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
    
class BIDSLoader(ProcessNode):
    outputs = ("bids_layout", )

    def _run(self, bids_directory : str, validate_bids : bool = False) -> tuple:
        return BIDSLayout(bids_directory, validate=validate_bids), 

class LoadBOLDData(ProcessNode):
    outputs = ("bold_img", "confounds_df", "events_df", "tr", "nr_measurements")
    
    def _run(self, bids_layout : BIDSLayout, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, load_events : bool = True) -> dict:
        # load bild data
        bold_img = image.load_img(
                        _checkBIDSQuery("BOLD", 
                            bids_layout.get(
                                return_type="file", 
                                    subject=subject, 
                                        session=session, 
                                            task=task, 
                                                run=run, 
                                                    space=space, 
                                                        suffix="bold", 
                                                            extension=".nii.gz")))
        
        # load in confounds 
        confounds_df = pd.read_csv(
                            _checkBIDSQuery("confounds", 
                                bids_layout.get(
                                    return_type="file", 
                                        subject=subject, 
                                            session=session, 
                                                task=task, 
                                                    run=run, 
                                                        suffix="timeseries", 
                                                            extension=".tsv")), sep='\t')
        
        # load in events
        if load_events:
            events_df = pd.read_csv(
                                _checkBIDSQuery("events", 
                                    bids_layout.get(
                                        return_type="file", 
                                            subject=subject, 
                                                session=session, 
                                                    task=task, 
                                                        run=run, 
                                                            suffix="events", 
                                                                extension=".tsv")), sep='\t')
        else:
            events_df = None

        # tr
        tr = bids_layout.get_metadata(bold_img.get_filename())["RepetitionTime"]
        # nr_meas
        nr_meas = bold_img.shape[3]
        
        return bold_img, confounds_df, events_df, tr, nr_meas


class LoadBidsImg(ProcessNode):
    outputs = ("bids_img", )
    
    def _run(self, bids_layout : BIDSLayout, subject : str = None, session : str = None, task : str = None, run : str = None, space : str = None, desc : str = None, suffix : str = None) -> dict:
        # check if is nifti file
        if _isNiftiFile(desc):
            bids_file = desc
        else:
            # if not, assume it is a description
            bids_file = _checkBIDSQuery(suffix, 
                        bids_layout.get(
                            return_type="file", 
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