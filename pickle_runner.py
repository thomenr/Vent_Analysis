import Vent_Analysis as vent
import numpy as np
import pyreadr # -- also need 'xarray' installed to use
import os
import nibabel as nib
import pickle

nifti_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/Archive'
files = os.listdir(nifti_path)
VDP = 0
k=0
for file in files:
    filepath = open(os.path.join(nifti_path,file), 'rb')
    pkl = pickle.load(filepath)
    Vent1 = vent.Vent_Analysis(pickle = pkl)
    #Vent1.runVDP()
    print(f'{files[k]} -- {pkl[2]['IRB']} --  Visit: {pkl[2]['visit'] } -- VDP: {pkl[2]['VDP']}')
    VDP = np.append(VDP,Vent1.VDP)
    k=k+1
    # Vent1.calculate_CI()
    # Vent1.CI
    #screenfile = f'{file}.png'
    #Vent1.screenShot(os.path.join('//umh.edu/data/Radiology/Xenon_Studies/Studies/ArchiveImages',screenfile))

for k in range(len(files)):
    print(f'{files[k]} -- {pkl[2]['IRB']} --  Visit: {pkl[2]['visit'] } -- VDP: {np.round(VDP[k+1],2)}')