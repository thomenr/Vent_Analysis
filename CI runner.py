import Vent_Analysis2 as vent
import numpy as np
import pyreadr # -- also need 'xarray' installed to use
import os
import nibabel as nib
import pickle

nifti_path = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/Niftis'
files = os.listdir(nifti_path)
file = files[0]
for file in files:
    A = nib.load(os.path.join(nifti_path,file)).get_fdata()
    #Vent1 = vent.Vent_Analysis(xenon_array = A['HP'].to_numpy(),mask_array = A['Mask'].to_numpy())
    Vent1 = vent.Vent_Analysis(xenon_array = A[:,:,:,0],mask_array = A[:,:,:,1])
    Vent1.calculate_VDP()
    Vent1.VDP
    Vent1.calculate_CI()
    Vent1.CI
    screenfile = f'{file}.png'
    Vent1.screenShot(os.path.join('//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/screenshots',screenfile))
    Vent1.pickleMe('c:/pirl/data/picklez')
    

    dataArray = Vent1.build4DdataArray();print('0')
    print(dataArray[0:3,0,0,1]);print('1')
    niImage = nib.Nifti1Image(dataArray, affine=np.eye(4));print('2')
    #niImage.header['pixdim'] = self.vox
    savepath = os.path.join(filepath,fileName + '_dataArray.nii');print('3')
    nib.save(niImage,savepath);print('4')
    print(f'\033[32mNifti HPvent array saved to {savepath}\033[37m');print('5')



# Set the path to the folder containing the pickle files
pkl_path = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/Pkls'
files = os.listdir(pkl_path)

# Iterate over all files in the pkl_path
for file in files:
    if file.endswith('.pkl'):  # Ensure only .pkl files are processed
        # Load the pickle file
        with open(os.path.join(pkl_path, file), 'rb') as f:
            data = pickle.load(f)  # Load the data from the pickle file

        # Extract relevant data from the loaded pickle object
        xenon_array = data[0][:,:,:,1]
        mask_array = data[0][:,:,:,2]

        # Initialize Vent_Analysis with the extracted data
        Vent1 = vent.Vent_Analysis(xenon_array=xenon_array, mask_array=mask_array)
        
        # Set voxel information
        Vent1.vox = [xenon_array[0, 0, 0], xenon_array[1, 0, 0], xenon_array[2, 0, 0]]

        # Perform Vent_Analysis operations
        Vent1.runVDP()
        Vent1.calculate_CI()

        # Save a screenshot
        screenshot_file = f'{file}.png'
        Vent1.screenShot(os.path.join('//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/screenshots', screenshot_file))

        # Export NIfTI
        nifti_output_path = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/CI/'
        nifti_file = f'{file}.nii'
        Vent1.exportNifti(filepath=nifti_output_path, fileName=nifti_file)
