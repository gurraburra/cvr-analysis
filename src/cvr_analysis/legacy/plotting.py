import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import image
import numpy as np
import sys
import json
from scipy.ndimage import affine_transform
from itertools import product

class IMGShow:
    def __init__(self, img, voxel_mask, settings, data_transform, *plot_data):
        # initate figure
        self.fig, self.axes = plt.subplots(2,2, figsize=(7,7))
        self.fig.tight_layout()
        self.fig.subplots_adjust(wspace=0, hspace=0, left=0, right=0.8, bottom=0.1, top=0.9)
        # set axis
        self.sagittal_ax = self.axes[0,0]
        self.coronal_ax = self.axes[0,1]
        self.axial_ax = self.axes[1,0]
        self.plot_ax = self.axes[1,1]
        # store data
        self.img = img
        self.voxel_mask = voxel_mask
        self.settings = settings
        self.data_transform = data_transform
        self.plot_data = plot_data
        # connect mpl
        self.fig.canvas.mpl_connect("button_press_event", self.buttonClick)
        # colormap
        self.settings["norm"] = mpl.colors.TwoSlopeNorm(vcenter = self.settings["vcenter"], vmin = self.settings["vmin"], vmax = self.settings["vmax"])
        del self.settings["vcenter"]
        del self.settings["vmin"]
        del self.settings["vmax"]
        self.scal_map = mpl.cm.ScalarMappable(norm=self.settings["norm"], cmap=self.settings["cmap"])
        # self.fig.set_facecolor("black")
        self.createColorbar()
        # set default position
        self.pos = [20,30,20]
        self.update()

    def update(self):
        for ax in self.axes.ravel():
            ax.cla()
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
        self.plot_ax.legend(loc = "lower left", title = f"pos: ({pos[0]},{pos[1]},{pos[1]}), val: {self.img[self.pos[0],self.pos[1],self.pos[2]]:.2f}")
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
            elif event.inaxes == self.coronal_ax:
                self.pos[0] = x
                self.pos[2] = y
            elif event.inaxes == self.sagittal_ax:
                self.pos[1] = x
                self.pos[2] = y
            self.update()
        except:
            pass

    def createColorbar(self):
        cbar_ax = self.fig.add_axes([0.86, 0.2, 0.05, 0.6])
        cbar_ax.yaxis.set_tick_params(color="black", labelcolor="black")
        self.fig.colorbar(self.scal_map, cax=cbar_ax)
        

def stand(sig):
    return (sig - np.nanmean(sig, axis = -1)[..., np.newaxis]) / np.nanstd(sig, axis = -1)[..., np.newaxis]

def maxMinNorn(sig):
    return sig / (np.nanmax(sig, axis = -1) - np.nanmin(sig, axis = -1))[..., np.newaxis]

norm_funcs = {"stand" : stand, "maxMinNorm" : maxMinNorn}

def showCVRAnalysisResult(analysis_file : str, img_desc = 'cvrAmplitude', img_transform = None, norm : str = "maxMinNorm", data_include = "bold+regressor+predictions", apply_transform = True, **custom_settings):
    # settings
    settings = {"cmap" : "RdYlBu_r", "vcenter" : 0, "vmin" : -1, "vmax" : 1, "aspect" : "equal", "origin" : "lower", 'interpolation' : 'antialiased'}
    settings.update(custom_settings)
    # norm_func
    norm_func = norm_funcs[norm]
    # folder preamble
    folder, analys_file = os.path.split(analysis_file)
    preamble = analys_file.split("_desc-analys_info")[0]
    # cvr file
    cvr_img = image.load_img(os.path.join(folder, preamble + f"_desc-{img_desc}_map.nii.gz"))
    # apply_transform 
    if apply_transform:
        # store all corner points
        points = []
        for x,y,z in product(*[[0,s-1] for s in cvr_img.shape]):
            points.append((x,y,z,1))
        points = np.array(points).T
        # transform using affine transform
        affine_points = np.linalg.matmul(cvr_img.affine, points)
        # find max and min
        min_loc = np.min(affine_points, axis = 1)[:3] 
        max_loc = np.max(affine_points, axis = 1)[:3]
        # find the voxel with smallest scale
        scale = np.abs(cvr_img.affine.diagonal()[:3]).min()
        # create new affine matrix
        affine = np.eye(4)
        affine[[0,1,2], [0,1,2]] = scale
        # set least value to min location
        affine[:3,3] = min_loc
        # make sure out shape cover whole field of view
        out_shape = np.ceil((max_loc - min_loc) / scale).astype(int)
        # create data transform matrix
        data_transform = np.linalg.matmul(np.linalg.inv(cvr_img.affine), affine)
        # resample cvr img
        cvr_img = image.resample_img(cvr_img, affine, out_shape, force_resample=True)
    else:
        data_transform = np.eye(4)
    # cvr data
    cvr_data = cvr_img.get_fdata()
    # mask
    # cvr_mask = np.abs(cvr_data) < 1e-10
    # mask data
    cvr_img_masked = cvr_data #  np.ma.masked_where(cvr_mask, cvr_data)
    if img_transform is not None:
        cvr_img_masked = img_transform(cvr_img_masked)
    # data
    data = []
    # spit data include
    data_include = data_include.split("+")
    # get timeshift data
    if "timeshift" in data_include:
        try:
            tshift_img = image.load_img(os.path.join(folder, preamble + "_desc-cvrTimeshift_map.nii.gz"))
            data.append((tshift_img.get_fdata(), "timeshift"))
        except:
            print("No timeshift img found")
    # get bold data
    if "bold" in data_include:
        try:
            bold_img = image.load_img(os.path.join(folder, preamble + "_desc-preproc_bold.nii.gz"))
            data.append((norm_func(bold_img.get_fdata()), "bold"))
        except:
            print("No bold img found")
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
            data.append((norm_func(predictions_img.get_fdata()), "prediction"))
        except:
            print("No prediction img found")
    # get predictions
    if "correlations" in data_include:
        try:
            correlations_img = image.load_img(os.path.join(folder, preamble + "_desc-correlations_map.nii.gz"))
            with open(os.path.join(folder, preamble + "_desc-data_info.json"), "r") as file:
                data_info = json.load(file)
            times = np.arange(0, correlations_img.shape[3]) * data_info["align-regressor-time-step"] + data_info["align-regressor-start-time"]
            corr = correlations_img.get_fdata()
            data.append(((times, norm_func(corr)), "x-correlation"))
        except:
            print("No correlation img found")
    # get voxel mask
    try:
        with open(os.path.join(folder, preamble + "_desc-data_info.json"), "r") as file:
            data_info = json.load(file)
        voxel_mask_img = image.resample_to_img(data_info['voxel-mask-file'], cvr_img, force_resample=True, interpolation="nearest")
        voxel_mask = np.ma.masked_where(voxel_mask_img.get_fdata(), voxel_mask_img.get_fdata())
        # voxel_mask = None
    except:
        voxel_mask = None
        print("Could not load voxel mask")

    img_show = IMGShow(cvr_img_masked, voxel_mask, settings, data_transform, 
                        *data)
    
    img_show.fig.suptitle(preamble)

    return img_show 

if __name__ == "__main__":
    img_show = showCVRAnalysisResult(sys.argv[1])
    # img_show.fig.show()
    plt.show()