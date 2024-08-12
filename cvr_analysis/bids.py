from process_control import ProcessNode
from bids import BIDSLayout
from nilearn import image, masking
import pandas as pd
import nibabel
import numpy as np
import re

class BIDSLoader(ProcessNode):
    outputs = ("bids_layout", )

    def _run(self, bids_directory : str, validate_bids : bool = False) -> tuple:
        return BIDSLayout(bids_directory, validate=validate_bids), 

    
class ImageLoader(ProcessNode):
    outputs = ("bold_img", "mask_img", "mask_description", "confounds_df", "events_df", "tr", "nr_measurements")
    
    def _run(self, bids_layout : BIDSLayout, subject : str, session : str = None, task : str = None, run : str = None, space : str = None, custom_mask : str = None, load_events : bool = True, remove_non_positive_values : bool = True) -> dict:
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
                                                        desc = 'brain',
                                                            suffix="mask", 
                                                                extension=".nii.gz", invalid_filters='allow'))
            # make sure mask and bold are in same space
            mask_img = image.resample_to_img(mask_file, bold_img, interpolation="nearest")
            mask_desc = "brain"
        elif custom_mask == "compute_epi_mask":
            mask_img = masking.compute_epi_mask(bold_img)
            mask_desc = custom_mask
        else:
            # make sure mask and bold are in same space
            mask_img = image.resample_to_img(custom_mask, bold_img, interpolation="nearest")
            mask_desc = re.search("desc-([^_]*)_", custom_mask).group(1)
        
        # remove negative voxels
        if remove_non_positive_values:
            mask_img = masking.intersect_masks([mask_img, image.math_img("np.all(img > 0, axis = -1)", img = bold_img)], threshold=1, connected=True)

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
        
        return bold_img, mask_img, mask_desc, confounds_df, events_df, tr, nr_meas

    def _checkBIDSQuery(self, file : str, files : list[str]) -> str:
        if len(files) == 0:
            raise ValueError(f"Could not find {file} file.")
        elif len(files) > 1:
            raise ValueError(f"Found multiple {file} files.")
        else:
            return files[0]
        
    
class DsegLoader(ProcessNode):
    outputs = ("labels_img",)
    
    def _run(self, bids_layout : BIDSLayout, subject : str, session : str = None, task : str = None, run : str = None, space : str = None, desc : str = None) -> dict:
        # load labels img
        labels_img = image.load_img(
                        self._checkBIDSQuery("BOLD", 
                            bids_layout.get(
                                return_type="file", 
                                    subject=subject, 
                                        session=session, 
                                            task=task, 
                                                run=run, 
                                                    space=space, 
                                                        desc=desc,
                                                            suffix="dseg", 
                                                                extension=".nii.gz")))
        return labels_img,

    def _checkBIDSQuery(self, file : str, files : list[str]) -> str:
        if len(files) == 0:
            raise ValueError(f"Could not find {file} file.")
        elif len(files) > 1:
            raise ValueError(f"Found multiple {file} files.")
        else:
            return files[0]
        