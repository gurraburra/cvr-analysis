import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import image
import numpy as np
import sys
from scipy.ndimage import spline_filter
import json

class IMGShow:
    def __init__(self, img, voxel_mask, settings, *plot_data):
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
        for data, label in self.plot_data:
            if data.ndim == 1:
                self.plot_ax.plot(data, label = label)
            elif data.ndim == 4:
                self.plot_ax.plot(data[self.pos[0], self.pos[1], self.pos[2]], label = label)
            else:
                raise ValueError("Incorrect dimensions")
        self.plot_ax.legend(loc = "lower left", title = f"val: {self.img[self.pos[0],self.pos[1],self.pos[2]]:.2f}")
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
            x, y = int(event.xdata), int(event.ydata)
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

def showCVRAnalysisResult(analysis_file : str, img_desc = 'cvrAmplitude', norm : str = "maxMinNorm", **custom_settings):
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
    # cvr data
    cvr_data = cvr_img.get_fdata()
    # mask
    cvr_mask = np.abs(cvr_data) < 1e-10
    # mask data
    cvr_img_masked = cvr_data #  np.ma.masked_where(cvr_mask, cvr_data)
    # data
    data = []
    # get bold data
    try:
        bold_img = image.load_img(os.path.join(folder, preamble + "_desc-preproc_bold.nii.gz"))
        data.append((norm_func(bold_img.get_fdata()), "bold"))
    except:
        print("No bold img found")
    # get aligned regressor data
    try:
        regressor_img = image.load_img(os.path.join(folder, preamble + "_desc-alignedRegressor_map.nii.gz"))
        data.append((norm_func(regressor_img.get_fdata()), "regressor"))
    except:
        print("No regressor img found")
    # get predictions
    try:
        predictions_img = image.load_img(os.path.join(folder, preamble + "_desc-predictions_bold.nii.gz"))
        data.append((norm_func(predictions_img.get_fdata()), "prediction"))
    except:
        print("No prediction img found")
    # get voxel mask
    try:
        with open(os.path.join(folder, preamble + "_desc-data_info.json"), "r") as file:
            data_info = json.load(file)
        voxel_mask = image.load_img(data_info['voxel-mask-file']).get_fdata()
        voxel_mask = np.ma.masked_where(voxel_mask, voxel_mask)
        # voxel_mask = None
    except:
        voxel_mask = None
        print("Could not load voxel mask")

    img_show = IMGShow(cvr_img_masked, voxel_mask, settings, 
                        *data)
    
    img_show.fig.suptitle(preamble)

    return img_show 

if __name__ == "__main__":
    img_show = showCVRAnalysisResult(sys.argv[1])
    # img_show.fig.show()
    plt.show()