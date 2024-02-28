from process_control import ProcessNode
from bids import BIDSLayout
from nilearn import image
import pandas as pd
import nibabel
import numpy as np

class BIDSLoader(ProcessNode):
    outputs = ("bids_layout", )

    def _run(self, bids_directory : str, validate_bids : bool = False) -> tuple:
        return BIDSLayout(bids_directory, validate=validate_bids), 

    
class ImageLoader(ProcessNode):
    outputs = ("bold_img", "mask_img", "confounds_df", "events_df", "tr", "nr_measurements")
    
    def _run(self, bids_layout : BIDSLayout, subject : str, session : str = None, task : str = None, run : str = None, space : str = None, custom_mask : str = None, load_events : bool = True) -> dict:
        # load bild data
        bold_img = image.load_img(
                        self._checkBIDSQuery("BOLD", 
                            bids_layout.get(
                                return_type="file", 
                                    subject=subject, 
                                        session=session, 
                                            task=task, 
                                                run=run, 
                                                    space=space, 
                                                        suffix="bold", 
                                                            extension=".nii.gz")))
        # load in mask file
        if custom_mask is None: # if no custom mask given, load from BIDS
            mask_file = self._checkBIDSQuery("mask", 
                            bids_layout.get(
                                return_type="file", 
                                    subject=subject, 
                                        session=session, 
                                            task=task, 
                                                run=run, 
                                                    space=space, 
                                                        suffix="mask", 
                                                            extension=".nii.gz"))
        else:
            mask_file = custom_mask
        # make sure mask and bold are in same space
        mask_img = image.resample_to_img(mask_file, bold_img, interpolation="nearest")
        # load in confounds 
        confounds_df = pd.read_csv(
                            self._checkBIDSQuery("confounds", 
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
                                self._checkBIDSQuery("events", 
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
        
        return bold_img, mask_img, confounds_df, events_df, tr, nr_meas

    def _checkBIDSQuery(self, file : str, files : list[str]) -> str:
        if len(files) == 0:
            raise ValueError(f"Could not find {file} file.")
        elif len(files) > 1:
            raise ValueError(f"Found multiple {file} files.")
        else:
            return files[0]
        
    