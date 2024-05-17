import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn import image
import json
import numpy as np
import sys

class IMGShow:
    def __init__(self, img, settings, *plot_data):
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
        self.settings = settings
        self.plot_data = plot_data
        # connect mpl
        self.fig.canvas.mpl_connect("button_press_event", self.buttonClick)
        # set default position
        self.pos = [20,30,20]
        self.update()
        # colormap
        norm = mpl.colors.Normalize(vmin = -self.settings["vmax"], vmax = self.settings["vmax"])
        self.scal_map = mpl.cm.ScalarMappable(norm=norm, cmap=self.settings["cmap"])
        # self.fig.set_facecolor("black")
        self.createColorbar()

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
        # plot ax
        for data, label in self.plot_data:
            if data.ndim == 1:
                self.plot_ax.plot(data, label = label)
            elif data.ndim == 4:
                self.plot_ax.plot(data[self.pos[0], self.pos[1], self.pos[2]], label = label)
            else:
                raise ValueError("Incorrect dimensions")
        self.plot_ax.legend(loc = "lower right")
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
        cbar_ax.yaxis.set_tick_params(color="white", labelcolor="white")
        self.fig.colorbar(self.scal_map, cax=cbar_ax)
        

def stand(sig):
    return (sig - np.nanmean(sig, axis = -1)[..., np.newaxis]) / np.nanstd(sig, axis = -1)[..., np.newaxis]

def showCVRAnalysisResult(analysis_file : str):
    # settings
    settings = {"cmap" : "RdYlBu_r", "vmin" : -1, "vmax" : 1, "aspect" : "equal", "origin" : "lower"}
    # folder preamble
    folder, analys_file = os.path.split(analysis_file)
    preamble = analys_file.split("desc-analys_info")[0]
    # cvr file
    cvr_img = image.load_img(os.path.join(folder, preamble + "desc-CVRAmplitude_map.nii.gz"))
    # mask cvr file
    cvr_mask = np.abs(cvr_img.get_fdata()) < 1e-10
    cvr_img_masked = np.ma.masked_where(cvr_mask, cvr_img.get_fdata())
    # data
    data = []
    # get bold data
    try:
        bold_img = image.load_img(os.path.join(folder, preamble + "desc-filteredBold_bold.nii.gz"))
        data.append((stand(bold_img.get_fdata()), "bold"))
    except:
        print("No bold img found")
    # get aligned regressor data
    try:
        regressor_img = image.load_img(os.path.join(folder, preamble + "desc-alignedRegressor_map.nii.gz"))
        data.append((stand(regressor_img.get_fdata()), "regressor"))
    except:
        print("No regressor img found")
    # get predictions
    try:
        predictions_img = image.load_img(os.path.join(folder, preamble + "desc-boldPredictions_map.nii.gz"))
        data.append((stand(predictions_img.get_fdata()), "prediction"))
    except:
        print("No prediction img found")

    img_show = IMGShow(cvr_img_masked, settings, 
                        *data)
    return img_show 

if __name__ == "__main__":
    img_show = showCVRAnalysisResult(sys.argv[1])
    # img_show.fig.show()
    plt.show()