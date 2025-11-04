import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import image
import numpy as np
import sys
import json
from scipy.ndimage import affine_transform
from itertools import product
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

def make_2x2_with_colorbar(ax=None, cbar_width=0.05, cbar_pad=0.02):
    """
    Create (or insert) a 2x2 grid of subplots with a single colorbar axis
    spanning both rows on the right.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None
        If None -> creates a new Figure with layout.
        If given -> replaces that axis with this layout in the same position.
    cbar_width : float
        Relative width of colorbar area.
    cbar_pad : float
        Padding between the main plots and colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : 2D numpy array of Axes (shape [2,2])
    cax : Axes for colorbar
    """

    if ax is None:
        # Make a new figure with overall gridspec
        fig = plt.figure(figsize=(8, 6))
        outer_gs = GridSpec(1, 1, figure=fig)
        sub_gs = outer_gs[0].subgridspec(2, 3, width_ratios=[1, 1, cbar_width],
                                         wspace=cbar_pad)
    else:
        # Replace the given axis with this layout
        fig = ax.figure
        parent_spec = ax.get_subplotspec()
        sub_gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=parent_spec,
                                         width_ratios=[1, 1, cbar_width],
                                         wspace=cbar_pad)
        ax.remove()

    # Create the 2x2 main axes
    axs = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            axs[i, j] = fig.add_subplot(sub_gs[i, j])

    # Create the colorbar axis spanning both rows
    cax = fig.add_subplot(sub_gs[:, 2])

    return fig, axs, cax

class IMGShow:
    def __init__(self, img, voxel_mask, bg_img, settings, data_transform, *plot_data, ax = None):
        # initate figure
        self.fig, self.axes, self.cbar_ax = make_2x2_with_colorbar(ax)#plt.subplots(2,2, )
        # self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0, hspace=0, left=0, right=0.8, bottom=0.1, top=0.9)
        # set axis
        self.sagittal_ax = self.axes[0,0]
        self.coronal_ax = self.axes[0,1]
        self.axial_ax = self.axes[1,0]
        self.plot_ax = self.axes[1,1]
        # store data
        self.img = img
        self.bg_img = bg_img
        self.voxel_mask = voxel_mask
        self.settings = settings
        self.data_transform = data_transform
        self.plot_data = plot_data
        # connect mpl
        self.fig.canvas.mpl_connect("button_press_event", self.buttonClick)
        # colormap
        cmap, norm = self.makeCmap(self.settings["cmap"], self.settings["vcenter"], self.settings["vmin"], self.settings["vmax"], self.settings["threshold"])
        # self.settings["norm"] = mpl.colors.TwoSlopeNorm(vcenter = self.settings["vcenter"], vmin = self.settings["vmin"], vmax = self.settings["vmax"])
        del self.settings["vcenter"]
        del self.settings["vmin"]
        del self.settings["vmax"]
        del self.settings["threshold"]
        self.settings["norm"] = norm
        self.settings["cmap"] = cmap
        self.scal_map = mpl.cm.ScalarMappable(norm=self.settings["norm"], cmap=self.settings["cmap"])
        # self.fig.set_facecolor("black")
        self.createColorbar()
        # set default position
        self.pos = ((np.array(img.shape) - 1) / 2).astype(int).tolist()
        # self linked imgs
        self.linked_imgs = []
        # update
        self.update()

    def update(self):
        for ax in self.axes.ravel():
            ax.cla()
        if self.bg_img is not None:
            self.sagittal_ax.imshow(self.bg_img[self.pos[0],:,:].T, cmap = "gray")
            self.coronal_ax.imshow(self.bg_img[:,self.pos[1],:].T, cmap = "gray")
            self.axial_ax.imshow(self.bg_img[:,:,self.pos[2]].T, cmap = "gray")
        # sagittal
        self.sagittal_ax.imshow(self.img[self.pos[0],:,:].T, **self.settings)
        self.sagittal_ax.axvline(self.pos[1], color = "black", alpha = 0.5)
        self.sagittal_ax.axhline(self.pos[2], color = "black", alpha = 0.5)
        # coronal
        self.coronal_ax.imshow(self.img[:,self.pos[1],:].T, **self.settings)
        self.coronal_ax.axvline(self.pos[0], color = "black", alpha = 0.5)
        self.coronal_ax.axhline(self.pos[2], color = "black", alpha = 0.5)
        # axial
        self.axial_ax.imshow(self.img[:,:,self.pos[2]].T, **self.settings)
        self.axial_ax.axvline(self.pos[0], color = "black", alpha = 0.5)
        self.axial_ax.axhline(self.pos[1], color = "black", alpha = 0.5)
        # voxel mask
        if self.voxel_mask is not None:
            self.sagittal_ax.imshow(self.voxel_mask[self.pos[0],:,:].T, cmap = 'Greys', aspect = self.settings["aspect"], origin = self.settings["origin"], interpolation = 'nearest')
            self.coronal_ax.imshow(self.voxel_mask[:,self.pos[1],:].T, cmap = 'Greys', aspect = self.settings["aspect"], origin = self.settings["origin"], interpolation = 'nearest')
            self.axial_ax.imshow(self.voxel_mask[:,:,self.pos[2]].T, cmap = 'Greys', aspect = self.settings["aspect"], origin = self.settings["origin"], interpolation = 'nearest')
        # plot ax
        pos = self.pos.copy()
        if self.data_transform is not None:
            pos.append(1)
            pos = np.array(pos)[:,None]
            pos = np.round(np.linalg.matmul(self.data_transform, pos), 0).astype(int)[:3,0]          
        for data, label in self.plot_data:
            if isinstance(data, tuple):
                times, data = data
            else:
                times = None
            if data.ndim == 1:
                if times is None:
                    self.plot_ax.plot(data, label = label)
                else:
                    self.plot_ax.plot(times, data, label = label)
            elif data.ndim == 3:
                if times is None:
                    self.plot_ax.axvline(data[pos[0], pos[1], pos[2]], label = label)
                else:
                    raise ValueError("Time cannot be defined fore 3D data")
            elif data.ndim == 4:
                if times is None:
                    self.plot_ax.plot(data[pos[0], pos[1], pos[2]], label = label)
                else:
                    self.plot_ax.plot(times, data[pos[0], pos[1], pos[2]], label = label)
            else:
                raise ValueError("Incorrect dimensions")
        if self.plot_ax.lines:
            val = self.img[self.pos[0],self.pos[1],self.pos[2]]
            if np.ma.is_masked(val):
                val_str = "--"
            else:
                val_str = f"{val:.2f}"
            self.plot_ax.legend(loc = "lower left", title = f"pos: ({pos[0]},{pos[1]},{pos[2]}), val: {val_str}")
        # self.plot_ax.set_aspect('equal')

        self.setAxis()
    
    def setAxis(self):
        for ax in (self.coronal_ax, self.axial_ax, self.sagittal_ax):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            size = max(ymax-ymin,xmax-xmin)
            xadd = (size - (xmax - xmin)) / 2
            yadd = (size - (ymax - ymin)) / 2
            ax.set_xlim((xmin - xadd,xmax + xadd))
            ax.set_ylim((ymin - yadd,ymax + yadd))
            ax.tick_params(left = False, right = False , labelleft = False , 
                        labelbottom = False, bottom = False) 
            ax.axis('off')
        self.plot_ax.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
        self.plot_ax.axis('off')
        
    
    def buttonClick(self,event):
        try:
            x, y = int(round(event.xdata,0)), int(round(event.ydata, 0))
            if event.inaxes == self.axial_ax:
                self.pos[0] = x
                self.pos[1] = y
                for l in self.linked_imgs:
                    l.pos[0] = x
                    l.pos[1] = y
            elif event.inaxes == self.coronal_ax:
                self.pos[0] = x
                self.pos[2] = y
                for l in self.linked_imgs:
                    l.pos[0] = x
                    l.pos[2] = y
            elif event.inaxes == self.sagittal_ax:
                self.pos[1] = x
                self.pos[2] = y
                for l in self.linked_imgs:
                    l.pos[1] = x
                    l.pos[2] = y
            self.update()
            for l in self.linked_imgs:
                l.update()
        except:
            pass

    def createColorbar(self):
        self.cbar_ax.yaxis.set_tick_params(color="black", labelcolor="black")
        self.fig.colorbar(self.scal_map, cax=self.cbar_ax)


    def makeCmap(self, base_cmap='coolwarm', vcenter=0, vmin=-5, vmax=5, threshold=1.0):
        if vmin is None:
            vmin = -vmax
        if vcenter is None:
            vcenter = (vmax + vmin) / 2
        if threshold is not None:
            # Sample base cmap
            n = 256
            base = get_cmap(base_cmap)
            norm_vals = np.linspace(0, 1, n)
            colors = base(norm_vals)
            # Compute normalized threshold position (0–1 scale)
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            # nfind upper/lower idx
            upper_tr_idx = np.argmin(np.abs(norm_vals - np.clip(norm(abs(threshold)), 0, 1)))
            lower_tr_idx = np.argmin(np.abs(norm_vals - np.clip(norm(-abs(threshold)), 0, 1)))
            # Set alpha = 0 in |x| < threshold
            colors[lower_tr_idx : upper_tr_idx + 1, -1] = 0.0
            cmap = LinearSegmentedColormap.from_list(f"{base.name}_transparent", colors, N=n)
            return cmap, norm
        else:
            return base_cmap, TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        
    def setLinkedImg(self, img, bi_directional = True):
        if img not in self.linked_imgs:
            self.linked_imgs.append(img)
        if bi_directional:
            if self not in img.linked_imgs:
                img.linked_imgs.append(self)
        

def stand(sig):
    return (sig - np.nanmean(sig, axis = -1)[..., np.newaxis]) / np.nanstd(sig, axis = -1)[..., np.newaxis]

def maxMinNorm(sig):
    return sig / (np.nanmax(sig, axis = -1) - np.nanmin(sig, axis = -1))[..., np.newaxis]

norm_funcs = {"stand" : stand, "maxMin" : maxMinNorm, None : lambda x : x}

def showCVRAnalysisResult(analysis_file : str, img_desc = 'cvrAmplitude', ax = None, bg_img = None, cvr_transform = None, norm_data : str = "maxMin", data_include = "bold+regressor+predictions", voxel_mask = False, radiological = True, **custom_settings):
    # settings
    settings = {"cmap" : "RdYlBu_r", "vcenter" : 0, "vmin" : -1, "vmax" : 1, "threshold" : None, "aspect" : "equal", "origin" : "lower", 'interpolation' : 'antialiased'}
    settings.update(custom_settings)
    # norm_func
    norm_func = norm_funcs[norm_data]
    # folder preamble
    folder, analys_file = os.path.split(analysis_file)
    preamble = analys_file.split("_desc-analys_info")[0]
    # cvr img
    if img_desc == "residual_ratio":
        # residuals_img = image.math_img("postproc - pred", postproc = os.path.join(folder, preamble + "_desc-postproc_bold.nii.gz"), pred = os.path.join(folder, preamble + "_desc-predictions_bold.nii.gz"))
        # moving_mean = voxelwise_centered_moving_average_scipy(residuals_img,window = 10,nan_policy="omit")
        # residuals_demean_var = image.math_img("np.nanvar(resid - moving, axis = -1)", resid = residuals_img, moving = moving_mean)
        # cvr_img = image.math_img("np.where(res_var > 0, np.nanvar(moving, axis=-1) / res_var, 0)", res_var = residuals_demean_var, moving = moving_mean)
        
        residuals_img = image.math_img("postproc - pred", postproc = os.path.join(folder, preamble + "_desc-postproc_bold.nii.gz"), pred = os.path.join(folder, preamble + "_desc-predictions_bold.nii.gz"))
        # ratio = lf_hf_metrics_4d(np.nan_to_num(residuals_img.get_fdata(), nan=0.0), 1/1.3, 1/30, return_fraction=False)
        flat = spectral_flatness_4d(np.nan_to_num(residuals_img.get_fdata(), nan=0.0), 1/1.3)
        cvr_img = image.new_img_like(residuals_img, flat)
    else:
        cvr_img = image.load_img(os.path.join(folder, preamble + f"_desc-{img_desc}_map.nii.gz"))
    # sotr inverse tranform going from real space to cvr space
    data_transform = np.linalg.inv(cvr_img.affine)
    if bg_img is not None:
        # resample CVR
        cvr_img = image.resample_to_img(cvr_img, bg_img, "nearest", force_resample=True, copy_header=True)
    # resample image to uniform scale using get_zooms
    new_affine = cvr_img.affine @ np.diag(1 / np.array(cvr_img.header.get_zooms() + (1,)))
    new_shape = (np.array(cvr_img.shape) * np.array(cvr_img.header.get_zooms())).astype(int)
    # radiological
    if radiological:
        mirror_x = np.diag([-1,1,1,1])
        mirror_x[0,3] = (cvr_img.shape[0] - 1)
        new_affine @= mirror_x
    cvr_img = image.resample_img(cvr_img, new_affine, new_shape, "nearest", force_resample=True, copy_header=True)
    if bg_img is not None:
        bg_img = image.resample_img(bg_img, new_affine, new_shape, "nearest", force_resample=True, copy_header=True)
        bg_data = bg_img.get_fdata()
    else:
        bg_data = None

    # update data transform
    data_transform @= new_affine

    # data_transform @= cvr_img.affine
    
    # cvr file
    # cvr_img = transformIMG(cvr_img)
    # cvr data
    cvr_data = cvr_img.get_fdata()
    # mask
    # cvr_mask = np.abs(cvr_data) < 1e-10
    # mask data
    cvr_data = cvr_data #  np.ma.masked_where(cvr_mask, cvr_data)
    if cvr_transform is not None:
        cvr_data = cvr_transform(cvr_data)
    # data
    data = []
    # spit data include
    data_include = data_include.replace(" ","").split("+")
    # get timeshift data
    if "timeshift" in data_include:
        try:
            tshift_img = image.load_img(os.path.join(folder, preamble + "_desc-cvrTimeshift_map.nii.gz"))
            data.append((tshift_img.get_fdata(), "timeshift"))
        except:
            print("No timeshift img found")
    # get bold data
    if "postproc" in data_include:
        try:
            postproc_img = image.load_img(os.path.join(folder, preamble + "_desc-postproc_bold.nii.gz"))
            data.append((norm_func(postproc_img.get_fdata()), "bold"))
        except:
            print("No postproc img found")
    # get aligned regressor data
    if "regressor" in data_include:
        try:
            regressor_img = image.load_img(os.path.join(folder, preamble + "_desc-alignedRegressor_map.nii.gz"))
            data.append((norm_func(regressor_img.get_fdata()), "regressor"))
        except:
            print("No regressor img found")
    # get predictions
    if "predictions" in data_include:
        try:
            predictions_img = image.load_img(os.path.join(folder, preamble + "_desc-predictions_bold.nii.gz"))
            data.append((norm_func(predictions_img.get_fdata()), "predictions"))
        except:
            print("No predictions img found")
    # get residuals
    if "residuals" in data_include:
        try:
            residuals_img = image.math_img("postproc - pred", postproc = os.path.join(folder, preamble + "_desc-postproc_bold.nii.gz"), pred = os.path.join(folder, preamble + "_desc-predictions_bold.nii.gz"))
            data.append((norm_func(residuals_img.get_fdata()), "residuals"))
            moving_mean = voxelwise_centered_moving_average_scipy(residuals_img,window = 30,nan_policy="omit")
            data.append((norm_func(moving_mean.get_fdata()), "moving_mean"))
        except:
            print("No predictions img found")
    # get predictions
    if "correlations" in data_include:
        try:
            correlations_img = image.load_img(os.path.join(folder, preamble + "_desc-correlations_map.nii.gz"))
            with open(os.path.join(folder, preamble + "_desc-data_info.json"), "r") as file:
                data_info = json.load(file)
            times = np.arange(0, correlations_img.shape[3]) * data_info["align-regressor-time-step"] + data_info["align-regressor-start-time"]
            corr = correlations_img.get_fdata()
            data.append(((times, norm_func(corr)), "welch power"))
        except:
            print("No correlation img found")
    # get predictions
    if "welch" in data_include:
        try:
            power_img = image.load_img(os.path.join(folder, preamble + "_desc-welch_power.nii.gz"))
            with open(os.path.join(folder, preamble + "_desc-welch_power.json"), "r") as file:
                data_info = json.load(file)
            freq = np.arange(0, power_img.shape[3]) * data_info["FreqSamplingStep"] + data_info["StartFreq"]
            power = power_img.get_fdata()
            data.append(((freq, norm_func(power)), "x-correlation"))
        except:
            print("No correlation img found")
    # get voxel mask
    try:
        if voxel_mask is not None:
            if voxel_mask == "non-zero":
                cvr_data = np.ma.masked_where(cvr_data == 0, cvr_data)
                # voxel_mask_img = (image.math_img("~np.isclose(img,0)", img = cvr_img))
            elif voxel_mask == "load":
                with open(os.path.join(folder, preamble + "_desc-data_info.json"), "r") as file:
                    data_info = json.load(file)
                voxel_mask_img = image.resample_to_img(data_info['voxel-mask-file'], cvr_img, force_resample=True, interpolation="nearest", copy_header=True)
                cvr_data = np.ma.masked_where(voxel_mask_img.get_fdata() == 0, cvr_data)
            # voxel_mask_data = voxel_mask_img.get_fdata()
            # voxel_mask = np.ma.masked_where(voxel_mask_data != 0, voxel_mask_data)
            # cvr_data = np.ma.masked_where(voxel_mask_data == 0, cvr_data)
        # else:
        #     voxel_mask = None
    except:
        # voxel_mask = None
        print("Could not load voxel mask")
    
    img_show = IMGShow(cvr_data, None, bg_data, settings, data_transform, *data, ax = ax)
    
    img_show.fig.suptitle(preamble)

    return img_show 

import numpy as np
from scipy.ndimage import uniform_filter1d

def voxelwise_centered_moving_average_scipy(residuals_img, window=11, mode="reflect", nan_policy="propagate"):
    """
    Centered moving average using SciPy's uniform_filter1d (no phase shift).

    Parameters
    ----------
    residuals_4d : np.ndarray
        4D residual map, shape (X, Y, Z, T).
    window : int
        Window size.
    mode : str
        Boundary handling for uniform_filter1d ("reflect", "nearest", "mirror", "wrap", "constant").
    nan_policy : {"propagate", "omit"}
        If "omit", temporarily fills NaNs by local means via two passes (mask + renormalization).

    Returns
    -------
    mav : np.ndarray
        Centered moving average with the same shape as input.
    """
    if residuals_img.ndim != 4:
        raise ValueError("residuals_4d must be 4D (X, Y, Z, T)")

    arr = residuals_img.get_fdata()

    if nan_policy == "propagate":
        # Simple centered filter; NaNs will propagate through neighborhoods
        return uniform_filter1d(arr, size=window, axis=-1, mode=mode, origin=0)

    elif nan_policy == "omit":
        # Compute local sums and counts, then divide to ignore NaNs
        valid = (~np.isnan(arr)).astype(arr.dtype)
        arr_filled = np.nan_to_num(arr, nan=0.0)

        local_sum   = uniform_filter1d(arr_filled, size=window, axis=-1, mode=mode, origin=0)
        local_count = uniform_filter1d(valid,     size=window, axis=-1, mode=mode, origin=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            mav = local_sum / local_count
        # Where count==0, set back to NaN
        mav[local_count == 0] = np.nan
        return image.new_img_like(residuals_img, mav)

    else:
        raise ValueError("nan_policy must be 'propagate' or 'omit'")


def voxelwise_variance_of_centered_moving_average_scipy(residuals_4d, window=11, **kwargs):
    mav = voxelwise_centered_moving_average_scipy(residuals_4d, window=window, **kwargs)
    return np.nanvar(mav, axis=-1)
import numpy as np
from scipy.signal import welch

def lf_hf_metrics_4d(
    data_4d,
    fs,
    f_c,
    nperseg=None,
    noverlap=None,
    detrend='constant',
    window='hann',
    average='mean',
    return_fraction=True,
):
    """
    Compute per-voxel LF/HF bandpower metrics for a 4D time series (X, Y, Z, T).

    Parameters
    ----------
    data_4d : np.ndarray
        4D array with shape (X, Y, Z, T).
    fs : float
        Sampling rate (Hz).
    f_c : float
        Cutoff frequency (Hz) separating low and high bands (0..Nyquist).
    nperseg, noverlap, detrend, window, average :
        Passed to scipy.signal.welch. Set nperseg to control frequency resolution.
        If None, SciPy chooses sensible defaults.
    return_fraction : bool
        If True, returns LF fraction = LF / (LF + HF); otherwise returns LF/HF ratio.

    Returns
    -------
    metric_3d : np.ndarray
        3D array (X, Y, Z) of LF fraction or LF/HF ratio per voxel.
        Voxels with zero HF power get np.inf for the ratio and 1.0 for the fraction.
    f : np.ndarray
        1D frequency vector used by Welch (Hz).
    """
    if data_4d.ndim != 4:
        raise ValueError("data_4d must be 4D with shape (X, Y, Z, T).")

    # Welch PSD along time axis for all voxels at once
    f, Pxx = welch(
        data_4d,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling='density',
        axis=-1,
        average=average,
    )
    # Pxx shape: (X, Y, Z, F), f shape: (F,)

    # Build masks for LF and HF bands
    # Include DC in LF; put the boundary in HF or LF consistently (here HF includes f >= f_c)
    lf_mask = (f >= 0) & (f < f_c)
    hf_mask = (f >= f_c)

    # Integrate band powers with trapezoid rule along frequency axis
    # np.trapz supports x as 1D; broadcasting along axis=-1 works.
    lf_power = np.trapezoid(Pxx[..., lf_mask], f[lf_mask], axis=-1)
    hf_power = np.trapezoid(Pxx[..., hf_mask], f[hf_mask], axis=-1)

    if return_fraction:
        total = lf_power + hf_power
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = lf_power / total
        # Where both bands are zero, set to NaN (no power at all)
        frac[(total == 0)] = np.nan
        return frac
    else:
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = lf_power / hf_power
        # If hf_power is zero but lf_power > 0 → inf; if both zero → NaN
        ratio[(hf_power == 0) & (lf_power == 0)] = np.nan
        return ratio


def spectral_flatness_4d(
    data_4d,
    fs,
    nperseg=None,
    noverlap=None,
    detrend='constant',
    window='hann',
    average='mean',
    eps=1e-30,
):
    """
    Per-voxel spectral flatness (Wiener entropy) for 4D time series (X, Y, Z, T).

    SFM ≈ 1 → white/noise-like (flat spectrum)
    SFM → 0 → structured/peaked (e.g., strong low-frequency or tones)

    Parameters
    ----------
    data_4d : np.ndarray, shape (X, Y, Z, T)
    fs : float
        Sampling rate (Hz).
    nperseg, noverlap, detrend, window, average :
        Passed to scipy.signal.welch.
    eps : float
        Floor to avoid log(0).

    Returns
    -------
    sfm_3d : np.ndarray
        3D array (X, Y, Z) of spectral flatness per voxel.
    f : np.ndarray
        1D frequency vector used by Welch (Hz).
    """
    if data_4d.ndim != 4:
        raise ValueError("data_4d must be 4D with shape (X, Y, Z, T).")

    f, Pxx = welch(
        data_4d,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling='density',
        axis=-1,
        average=average,
    )  # Pxx: (X, Y, Z, F)

    # Geometric mean / arithmetic mean along frequency axis
    Pxx_safe = np.maximum(Pxx, eps)
    gm = np.exp(np.mean(np.log(Pxx_safe), axis=-1))
    am = np.mean(Pxx, axis=-1)  # allow zeros here; gm floor already handled
    with np.errstate(invalid='ignore', divide='ignore'):
        sfm = gm / am
    # If spectrum is entirely zero at a voxel (degenerate), set NaN
    sfm[(am == 0)] = np.nan
    return sfm

if __name__ == "__main__":
    img_show = showCVRAnalysisResult(sys.argv[1])
    # img_show.fig.show()
    plt.show()