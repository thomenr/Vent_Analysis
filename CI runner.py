import Vent_Analysis as vent
import numpy as np
import pyreadr # -- also need 'xarray' installed to use
import os
import nibabel as nib

nifti_path = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/Niftis'
files = os.listdir(nifti_path)

for file in files:
    A = nib.load(os.path.join(nifti_path,file)).get_fdata()
    #Vent1 = vent.Vent_Analysis(xenon_array = A['HP'].to_numpy(),mask_array = A['Mask'].to_numpy())
    Vent1 = vent.Vent_Analysis(xenon_array = A[:,:,:,0],mask_array = A[:,:,:,1])
    Vent1.vox = [A[0,0,0,0],A[1,0,0,0],A[2,0,0,0]]
    Vent1.runVDP()
    Vent1.VDP
    Vent1.calculate_CI()
    Vent1.CI
    screenfile = f'{file}.png'
    Vent1.screenShot(os.path.join('//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/screenshots',screenfile))
    Vent1.exportNifti(filepath = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/CI/',fileName=f'{file}.nii')