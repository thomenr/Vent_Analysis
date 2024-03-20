## -- PIRL Ventilation Image Analysis Pipeline, RPT, 3/19/2024, version 240319_RPT -- ##
import numpy as np
import os
from scipy.signal import medfilt2d # ------ for calculateVDP
import time # ----------------------------- for calculateVDP
import SimpleITK as sitk # ---------------- for N4 Bias Correection
import CI # ------------------------------- for calculateCI
import mapvbvd # -------------------------- for process_Raw
from sys import getsizeof # --------------- To report twix object size
import tkinter as tk # -------------------- GUI stuffs
from tkinter import filedialog # ---------- for openSingleDICOM and openDICOMFolder
import pydicom as dicom # ----------------- for openSingleDICOM and openDICOMFolder
from matplotlib import pyplot as plt # ---- for makeSlide
import skimage.util # --------------------- for image montages
import nibabel as nib # ------------------- for Nifti stuffs
import PySimpleGUI as sg # ---------------- for GUI stuffs
from PIL import Image, ImageTk # ---------- for arrayToImage conversion

#------------------------------------------------------------------------------------
# ----------- VENTILATION ANALYSIS CLASS DEFINITION ---------------------------------
#------------------------------------------------------------------------------------
class Vent_Analysis:
    """Performs complete VDP analysis: N4Bias correction, normalization,
        defect calculation, and VDP calculation.
    INPUTS: 
        HPvent - 3D array of ventilation image stack
        mask - 3D array of lung segmentation for HPvent (must match HPvent shape)
    CALCULATED ATTRIBUTES: 
        version - Date and author of most recent Vent_Analysis update
        N4HPvent - N4 bias corrected ventilation array
        normMeanHPvent - ventilation array normalized to signal mean (mean-anchored)
        norm95HPvent - ventilation array normalized to signal 95th percentile (linear binning)
        defectArray - binary array of defect voxels (using mean-anchored array)
        VDP - the ventilation defect percentage using 60% treshold by default (using mean-anchored array)
        CIarray - the Cluster Index Array given the defectArray and vox
        CI - the 95th percentile Cluster Value
    METHODS:
        __init__ - Opens the HP Vent and mask dicoms into self.HPvent, and self.mask
        openSingleDICOM - opens a single 3D dicom file (for vent images)
        openDICOMfolder - opens all 2D dicoms in a folder (for mask images) into 3D array
        pullDICOMHeader - pulls useful DICOM header info into self variables
        runVDP - creates N4HPvent, defectarray (mean-anchored) and VDP
        calculateCI - calculates CI (imports CI functions)
        process_RAW - process the corresponding TWIX file associated
    """
    def __init__(self,dicom_path=None,mask_dir=None):
        self.version = '240319_RPT' # - update this when changes are made!! - #
        self.ds, self.HPvent = self.openSingleDICOM(dicom_path)
        self.pullDICOMHeader()
        _, self.mask = self.openDICOMfolder(mask_dir)
        print(f'Opened DICOM of shape {self.HPvent.shape} and MASK of shape {self.mask.shape}')
        self.mask_border = self.calculateBorder(self.mask)
        print(f'Calculated mask_border of shape {self.mask_border.shape}')

    def openSingleDICOM(self,dicom_path):        
        if dicom_path is None:
            root = tk.Tk()
            root.withdraw()
            print('\033[94mSelect the DICOM ventilation file...\033[37m')
            dicom_path = tk.filedialog.askopenfilename()
            ds = dicom.dcmread(dicom_path,force=True)
        else:
            ds = dicom.dcmread(dicom_path)
        HPvent = ds.pixel_array
        HPvent = np.transpose(HPvent,(1,2,0))
        print(f'\033[94mI opened a DICOM of shape {HPvent.shape}\033[37m')
        return ds, HPvent


    def openDICOMfolder(self,maskFolder):  
        from tkinter import filedialog
        if maskFolder is None:
            print('\033[94mSelect the mask folder...\033[37m')
            maskFolder = tk.filedialog.askdirectory()
        dcm_filelist = [f for f in os.listdir(maskFolder) if f.endswith('.dcm')]
        ds = dicom.dcmread(os.path.join(maskFolder,dcm_filelist[0]))
        mask = np.zeros((ds.pixel_array.shape[0],ds.pixel_array.shape[1],len(dcm_filelist)))
        for f,k in zip(dcm_filelist,range(len(dcm_filelist))):
            ds = dicom.dcmread(os.path.join(maskFolder,f))
            mask[:,:,k] = ds.pixel_array
        print(f'\033[94mI built a mask of shape {mask.shape}\033[37m')
        return ds, mask

    def pullDICOMHeader(self):
        self.PatientName = self.ds.PatientName
        self.PatientAge = self.ds.PatientAge
        self.PatientBirthDate = self.ds.PatientBirthDate
        self.PatientSize = self.ds.PatientSize
        self.PatientWeight = self.ds.PatientWeight
        self.PatientSex = self.ds.PatientSex
        self.StudyDate = self.ds.StudyDate
        self.StudyTime = self.ds.StudyTime
        self.SeriesTime = self.ds.SeriesTime
        for k in range(100):
            try:
                self.vox = self.ds[0x5200, 0x9230][k]['PixelMeasuresSequence'][0].PixelSpacing
                break
            except:
                print(k)
                if k == 99:
                    print('Pixel Spacing not in correct place, please enter each dimension...')
                    self.vox = [float(input()),float(input())]

        try:
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(self.ds.SpacingBetweenSlices)]
        except:
            print('Slice spacing not in correct position. Please enter manually:')
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(input())]

    def calculateBorder(self,A):
        border = np.zeros(A.shape)
        for k in range(A.shape[2]):
            x = np.gradient(A[:,:,k].astype(float))
            border[:,:,k] = (x[0]!=0)+(x[1]!=0)
        return border

    def runVDP(self):
        self.SNR = self.calculate_SNR(self.HPvent,self.mask) ## -- SNR of DICOM HP, not Raw, nor N4
        self.N4HPvent = self.N4_bias_correction(self.HPvent,self.mask)
        self.normMeanHPvent  = self.normalize_mean(self.N4HPvent,self.mask)
        self.norm95HPvent = self.normalize_95th(self.N4HPvent,self.mask)
        self.defectArray, self.defectBorder = self.calculateDefectArray(self.normMeanHPvent,self.mask,0.6)
        self.VDP = 100*np.sum(self.defectArray)/np.sum(self.mask)

    def calculate_CI(self):
        self.CIarray = CI.calculate_CI(self.defectArray,self.vox)
        CVlist = np.sort(self.CIarray[self.defectArray==1])
        index95 = int(0.95*len(CVlist))
        self.CI = CVlist[index95]
        #return self.CIarray, self.CI

    def exportNifti(self,filepath=None,fileName = None):
        if filepath == None:
            print('\033[94mWhere would you like to save your Niftis?\033[37m')
            filepath = tk.filedialog.askdirectory()

        if fileName == None:
            fileName = str(self.PatientName).replace('^','_')

        try:
            dataArray = self.build4DdataArray()
            niImage = nib.Nifti1Image(dataArray, affine=np.eye(4))
            nib.save(niImage,os.path.join(filepath,fileName + '_dataArray.nii'))
        except:
            print('\033[31mCould not build 4D HPvent mask Nifti\033[37m')

        try:
            niImage = nib.Nifti1Image(self.raw_HPvent, affine=np.eye(4))
            nib.save(niImage,os.path.join(filepath,fileName + '_raw_HPvent.nii'))
            print('\033[92mraw_HPvent Nifti saved\033[37m')
        except:
            print('\033[31mNifti Export of raw_HPvent failed\033[37m')

    def build4DdataArray(self):
        ''' Our arrays are: HPvent [0], mask  [1], N4HPvent [2], defectArray [3], CIarray [4]'''
        dataArray = np.zeros((self.HPvent.shape[0],self.HPvent.shape[1],self.HPvent.shape[2],5))
        dataArray[:,:,:,0] = self.HPvent
        dataArray[:,:,:,1] = self.mask
        try:
            dataArray[:,:,:,2] = self.N4HPvent
        except:
            print('\033[31mN4HPvent does not exist and was not added\033[37m')
        try:
            dataArray[:,:,:,3] = self.defectArray
        except:
            print('\033[31mdefectArray does not exist and was not added\033[37m')
        try:
            dataArray[:,:,:,4] = self.CIarray
        except:
            print('\033[31mCIarray does not exist and was not added\033[37m')
        return dataArray
    
    def buildMetadata(self):
        metadata = {
            'version': str(self.version),
            'DICOMPatientName': str(self.PatientName),
            'DICOMPatientAge': str(self.PatientAge),
            'DICOMPatientBirthDate': str(self.PatientBirthDate),
            'DICOMPatientHeight': str(self.PatientSize),
            'DICOMPatientWeight': str(self.PatientWeight),
            'DICOMPatientSex': str(self.PatientSex),
            'DICOMStudyDate': str(self.StudyDate),
            'DICOMStudyTime': str(self.StudyTime),
            'DICOMSeriesTime': str(self.SeriesTime),
            'DICOMVoxelSize': str(self.vox),
            }
        try:
            metadata['TWIXDateTime'] = str(self.scanDateTime)
            metadata['TWIXProtocol'] = str(self.protocolName)
        except:
            print('Could not add TWIX params to metadata. Have you run process_RAW() yet?')
        return metadata

    def process_RAW(self,filepath=None):
        if filepath == None:
            print('\033[94mSelect the corresponding RAW data file (Siemens twix)...\033[37m\n')
            filepath = tk.filedialog.askopenfilename()
    
        self.raw_twix = mapvbvd.mapVBVD(filepath)
        #self.subjectID = self.raw_twix.hdr.Config['PatientID']
        self.scanDateTime = self.raw_twix.hdr.Config['PrepareTimestamp']
        self.protocolName = self.raw_twix.hdr.Meas['tProtocolName']
        # self.age = self.raw_twix.hdr.Protocol['flPatientAge']
        # self.sex = self.raw_twix.hdr.Protocol['lPatientSex']
        self.raw_twix.image.squeeze = True
        self.raw_K = self.raw_twix.image['']
        self.raw_HPvent = np.zeros((self.raw_K.shape)).astype(np.complex128)
        for k in range(self.raw_K.shape[2]):
            self.raw_HPvent[:,:,k] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.raw_K[:,:,k])))
        self.raw_HPvent = np.transpose(self.raw_HPvent,(1,0,2))[:,::-1,:]

    def N4_bias_correction(self,HPvent, mask):
        start_time = time.time()
        print('Performing Bias Correction...')

        # Convert NumPy arrays to SimpleITK images
        image = sitk.GetImageFromArray(HPvent.astype(np.float32))
        mask = sitk.GetImageFromArray(mask.astype(np.float32))

        #Cast to correct format for SimpleITK
        image = sitk.Cast(image, sitk.sitkFloat32)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        #Run Bias Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image, mask)
        corrected_HPvent = sitk.GetArrayFromImage(corrected_image)
        print(f'Bias Correction Completed in {np.round(time.time()-start_time,2)} seconds')
        return corrected_HPvent

    def normalize_mean(self,ventilation_array,mask):
        '''returns HPvent array normalized to the whole-lung signal mean'''
        normHPvent = np.divide(ventilation_array,np.mean(ventilation_array[mask>0]))
        return normHPvent
    
    def calculate_SNR(self,A,FOVbuffer=20,manualNoise = False):
        '''Calculates SNR using all voxels in the mask as signal, and all 
        voxels oustide the mask bounding box as noise. Can also be done manually if desired'''
        signal = A[self.mask>0]
        if not manualNoise:
            noisemask = np.ones(self.mask.shape)
            FOVbuffer = 20
            rr = (np.sum(np.sum(self.mask,axis = 2),axis = 1)>0)*(list(range(self.mask.shape[0])))
            cc = (np.sum(np.sum(self.mask,axis = 0),axis = 1)>0)*(list(range(self.mask.shape[1])))
            cc = np.arange(np.min(cc[cc>0]),np.max(cc))
            ss = (np.sum(np.sum(self.mask,axis = 1),axis = 0)>0)*(list(range(self.mask.shape[2])))
            noisemask[np.ix_(rr,cc,ss)] = 0
            noisemask[:FOVbuffer,:,:] = 0
            noisemask[(noisemask.shape[0]-FOVbuffer):,:,:] = 0
            noise = A[noisemask==1]
        else:
            sub_array = hpg.get_subarray(self.HPvent[:,:,int(self.HPvent.shape[2]/2)])
            noise = sub_array['A'].flatten()
        SNR = (np.mean(signal)-np.mean(noise))/np.std(noise)
        self.SNR = SNR
        return SNR
    
    def calculateDefectArray(self,ventilation_array,mask,thresh):
        '''given a threshold (specified as a fraction of the whole-lung signal mean) create the 3D binary defect array'''
        defectArray = np.zeros(ventilation_array.shape)
        for k in range(mask.shape[2]):
            defectArray[:,:,k] = medfilt2d((ventilation_array[:,:,k]<thresh)*mask[:,:,k])
        defectBorder = self.calculateBorder(defectArray)
        return (defectArray==1), (defectBorder==1)

    def normalize_95th(self,ventilation_array,mask):
        voxlist = ventilation_array[mask>0]
        voxlist.sort()
        norm95HPvent = np.divide(ventilation_array,voxlist[int(0.95*len(voxlist))])
        norm95HPvent[norm95HPvent>1] = 1
        norm95HPvent = norm95HPvent*255
        norm95HPvent = norm95HPvent.astype(np.uint8)
        return norm95HPvent
    
    def makeSlide(self,A):
        plt.imshow(skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], padding_width=1, fill=0),cmap='gray')
        plt.show()

    def array3D_to_montage2D(self,A):
        #print(f'Array 3D to montage 2D called with array of shape {A.shape}...')
        return skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], grid_shape = (1,A.shape[2]), padding_width=0, fill=0)

    def crop_to_data(self, A, border=0):
        '''Not working currently, not sure why (240308_RPT)'''
        rr = np.where(np.sum(A, axis=1) > 0)[0] + 1
        cc = np.where(np.sum(A, axis=2) > 0)[0] + 1
        ss = np.where(np.sum(A, axis=0) > 0)[0] + 1
        print('---------------------------------------------------WHERE MASK SUM IS > 0 ---------------------------------')
        print(rr)
        print(cc)
        print(ss)

        if border > 0:
            for _ in range(border):
                rr = np.concatenate(([np.min(rr) - 1], rr, [np.max(rr) + 1]))
                cc = np.concatenate(([np.min(cc) - 1], cc, [np.max(cc) + 1]))

        print('---------------------------------------------------REMOVE values greater than ---------------------------------')
        rr = rr[(rr > 0) & (rr < A.shape[0])]
        cc = cc[(cc > 0) & (cc < A.shape[1])]
        ss = ss[(ss > 0) & (ss < A.shape[2])]
        print('---')
        print(rr)
        print(cc)
        print(ss)
        A = A[rr-1, cc-1, ss-1]
        print('Calculated A using rr,cc,ss')

        return {"A": A, "rr": rr, "cc": cc, "ss": ss}
            
            
    def __str__(self) -> str:
        '''What do you want the class to print for you when you check it? Add that here!'''
        string = (f'\033[35mVent_Analysis\033[37m class object version \033[94m{self.version} \033[37m\n')

        string = string + (f'\033[92mSUBJECT DATA --\033[37m\n')

        try:
            string = string + (f'  \033[96mSubject\033[37m: {self.PatientName}\n')
        except:
            string = string + (f'  \033[96mSubject\033[37m: \033[32mnot yet run\033[37m\n')
        
        try:
            string = string + (f'  \033[96mAge\033[37m: {self.PatientAge}\n')
        except:
            string = string + (f'  \033[96mAge\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mSex\033[37m: {self.PatientSex}\n')
        except:
            string = string + (f'  \033[96mSex\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mBirth Date\033[37m: {self.PatientBirthDate}\n')
        except:
            string = string + (f'  \033[96mBirth Date\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mHeight\033[37m: {self.PatientSize}m ({int(np.floor(self.PatientSize*3.28084))}ft {int(12*(self.PatientSize*3.28084 - (self.PatientSize*3.28084 // 1)))}in)\n')
        except:
            string = string + (f'  \033[96mHeight\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mWeight\033[37m: {self.PatientWeight}kg ({int(self.PatientWeight*2.2046)} lbs)\n')
        except:
            string = string + (f'  \033[96mWeight\033[37m: \033[32mnot yet run\033[37m\n')


        string = string + (f'\033[92mDICOM DATA --\033[37m\n'
                f'  \033[96mHPvent\033[37m array shape: {self.HPvent.shape}\n'
                f'  \033[96mMask\033[37m lung vol: {np.round(np.sum(self.mask>0)*np.prod(self.vox)/1000/1000,1)} L\n')

        try:
            string = string + (f'  \033[96mDefect\033[37m volume: {np.round(np.sum(self.defectArray>0)*np.prod(self.vox)/1000/1000,1)} L\n')
        except:
            string = string + (f'  \033[96mDefect\033[37m volume: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mVoxel Size\033[37m: {self.vox}\n')
        except:
            string = string + (f'  \033[96mVoxel Size\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mSNR\033[37m = {np.round(self.SNR,2)}\n')
        except:
            string = string + (f'  \033[96mSNR\033[37m = \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mVDP\033[37m = {np.round(self.VDP,2)}\n')
        except:
            string = string + (f'  \033[96mVDP\033[37m = \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mCI\033[37m = {np.round(self.CI,2)}\n')
        except:
            string = string + (f'  \033[96mCI\033[37m = \033[32mnot yet run\033[37m\n')

        string = string + (f'\033[92mRAW DATA --\033[37m\n')

        try:
            string = string + (f'  \033[96mkSpace shape\033[37m: {self.raw_K.shape}\n')
        except:
            string = string + (f'  \033[96mkSpace shape\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mScan Date/Time\033[37m: {self.scanDateTime}\n')
        except:
            string = string + (f'  \033[96mScan Date/Time\033[37m: \033[32mnot yet run\033[37m\n')

        try:
            string = string + (f'  \033[96mProtocol\033[37m: {self.protocolName}\n')
        except:
            string = string + (f'  \033[96mProtocol\033[37m: \033[32mnot yet run\033[37m\n')


        return string

    def __repr__(self):
        '''This is necessary to have the default printout be the dunder str above I think.'''
        return str(self)


### ------------------------------------------------------------------------------------------------ ###
### ---------------------------------------- HELPER FUNCTIONS -------------------------------------- ###
### ------------------------------------------------------------------------------------------------ ###

def arrayToImage(A,size):
    imgAr = Image.fromarray(A.astype(np.uint8))
    imgAr = imgAr.resize(size)
    image = ImageTk.PhotoImage(image=imgAr)
    return(image)


normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))

def colorBinary(A,B):
    A = normalize(A)
    new = np.zeros((A.shape[0],A.shape[1],3))
    new[:,:,0] = A*(B==0) + B
    new[:,:,1] = A*(B==0)
    new[:,:,2] = A*(B==0)
    return new*255



def extract_attributes(attr_dict, parent_key='', sep='_'):
    """
    Recursively extract all attributes and subattributes from a nested dictionary and compiles into flat dictionary.
    
    Args:
    - attr_dict (dict): The attribute dictionary to extract from.
    - parent_key (str): The base key to use for building key names for subattributes.
    - sep (str): The separator to use between nested keys.
    
    Returns:
    - dict: A flat dictionary with all attributes and subattributes.
    """
    items = []
    for k, v in attr_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # If the value is a dictionary, recurse
            items.extend(extract_attributes(v, new_key, sep=sep).items())
        else:
            # Otherwise, add the attribute to the items list
            items.append((new_key, v))
    return dict(items)


### ------------------------------------------------------------------------------------------------ ###
### ---------------------------------------- MAIN SCRIPT ------------------------------------------- ###
### ------------------------------------------------------------------------------------------------ ###

image_box_size = 70
if __name__ == "__main__":
    import PySimpleGUI as sg
    import json
    import h5py
    version = '240319_RPT'
    ARCHIVE_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/H5/'
    sg.theme('Default1')
    PIRLlogo = 'C:/PIRL/HPG/PIRLlogo.png'
    path_label_column = [[sg.Text('Path to Ventilation DICOM:')],[sg.Text('Path to Mask Folder:')],[sg.Text('Path to Twix:')]]
    path_column = [[sg.InputText(key='path1',default_text='C:/PIRL/data/MEPOXE0037/XENON',size=(200,200))],
                   [sg.InputText(key='path2',default_text='C:/PIRL/data/MEPOXE0037/Mask',size=(200,200))],
                   [sg.InputText(key='path3',default_text='C:/PIRL/data/MEPOXE0037/meas_MID00146_FID56770_6_FLASH_gre_hpg_2201_SliceThi_10.dat',size=(200,200))]]
    
    IRB_select_column = [
                    [sg.Radio('GenXe','IRB',key='genxeRadio',enable_events=True)],
                    [sg.Radio('Mepo','IRB',key='mepoRadio',enable_events=True)],
                    [sg.Radio('Clinical','IRB',key='clinicalRadio',enable_events=True)]]
    genxe_info_column = [[sg.Text('General Xenon ID:'),sg.InputText(default_text='0000',size=(10,10),key='genxeID')],
                           [sg.Checkbox('PreAlbuterol',default=False,key='prealb'),sg.Checkbox('PostAlbuterol',default=False,key='postalb')],
                           [sg.Checkbox('PreSildenafil',default=False,key='presil'),sg.Checkbox('PostSildenafil',default=False,key='postsil')],
                           ]
    mepo_info_column = [[sg.Text('Mepo ID:'),sg.InputText(default_text='0000',size=(10,10),key='mepoID')],
                        [sg.Text('Mepo Subject #:    '),sg.InputText(default_text='0',size=(10,10),key='meposubjectnumber')],
                        [sg.Text('Visit:    '),sg.Radio('Baseline','mepo_visit',key='mepoVisit1'),sg.Radio('4-week','mepo_visit',key='mepoVisit2'),sg.Radio('12-week','mepo_visit',key='mepoVisit3')],
                        [sg.Radio('PreAlbuterol','mepoalbuterol',key='prealb_mepo'),sg.Radio('PostAlbuterol','mepoalbuterol',key='postalb_mepo')],
                        ]
    clinical_info_column = [[sg.Text('Clinical Subject Initials:'),sg.InputText(default_text='',size=(10,10),key='clinicalID')],
                           [sg.Text('Visit #:    '),sg.InputText(default_text='0',size=(10,10),key='clinicalvisitnumber')],
                           [sg.Radio('Single Session','clinicalalbuterol',key='singlesession'),
                            sg.Radio('PreAlbuterol','clinicalalbuterol',key='prealb_clin'),
                            sg.Radio('PostAlbuterol','clinicalalbuterol',key='postalb_clin')],
                           ]
    dose_info_column = [[sg.Text('DE [mL]:'),sg.InputText(key='DE',size=(10,10))],
                           [sg.Text('FEV1 [%]: '),sg.InputText(key='FEV1',size=(10,10))],
                           [sg.Text('FVC [%]: '),sg.InputText(key='FVC',size=(10,10))],
                           ]
    
    patient_data_column = [[sg.Text('Subject:                               ',key='subject',pad=(0,0))],
                           [sg.Text('Study Date:',key='studydate',pad=(0,0))],
                           [sg.Text('Study Time:',key='studytime',pad=(0,0))],
                           [sg.Text('Twix Date:',key='twixdate',pad=(0,0))],
                           [sg.Text('Protocol:',key='twixprotocol',pad=(0,0))],
                           [sg.Text('Age:',key='age',pad=(0,0))],
                           [sg.Text('Sex:',key='sex',pad=(0,0))],
                           [sg.Text('DOB:',key='dob',pad=(0,0))],
                           [sg.Text('Height:',key='height',pad=(0,0))],
                           [sg.Text('Weight:',key = 'weight',pad=(0,0))]]
    dicom_data_column = [[sg.Text('DICOM Voxel Size:                                ',key = 'vox',pad=(0,0))],
                         [sg.Text('SNR:',key = 'snr',pad=(0,0))],
                         [sg.Text('VDP:',key = 'vdp',pad=(0,0))],
                         [sg.Text('Ventilation Array Shape:',key='ventarrayshape',pad=(0,0))],
                         [sg.Text('Mask Lung Vol:',key='masklungvol',pad=(0,0))],
                         [sg.Text('Defect Volume:',key='defectvolume',pad=(0,0))],
                         [sg.Text('CI:',key='ci',pad=(0,0))]]
    image_column = [[sg.Image(key='-RAWIMAGE-')],
                    [sg.Image(key='-N4IMAGE-')],
                    [sg.Image(key='-DEFECTIMAGE-')],
                    [sg.Image(key='-TWIXIMAGE-')]]

    layout = [
        [sg.Image(PIRLlogo),sg.Text(f'version {version}'),sg.Text('         User:'),sg.InputText(key='userName',size=(10,1),enable_events=False),sg.Button('-',key='minus'),sg.Button('+',key='plus')],
        [sg.HorizontalSeparator()],
        [sg.Column(path_label_column),sg.Column(path_column)],
        [sg.Button('Load from Paths', key='-INITIALIZE-'),sg.Button('Calculate VDP', key='-CALCVDP-'),sg.Button('Calculate CI', key='-CALCCI-'),sg.Button('Import TWIX', key='-RUNTWIX-')],  
        [sg.HorizontalSeparator()],
        [sg.Column(IRB_select_column),
         sg.Column(clinical_info_column,key='clinicalInputs',visible=False),
         sg.Column(genxe_info_column,key='genxeInputs',visible=False),
         sg.Column(mepo_info_column,key='mepoInputs',visible=False),
         sg.Column(dose_info_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(patient_data_column),sg.VSeperator(),sg.Column(dicom_data_column),sg.VSeperator(),sg.Column(image_column)],  
        [sg.Text('',key = '-STATUS-')],
        [sg.InputText(key='exportpath',default_text='C:/PIRL/data/MEPOXE0037/',size=(200,200))],
        [sg.Button('Export Data',key='-EXPORT-'),sg.Checkbox('Copy H5 to Archive',default=True,key='-ARCHIVE-')]
    ]

    window = sg.Window(f'PIRL Ventilation Analysis -- {version}', layout, return_keyboard_events=True, margins=(0, 0), finalize=True, size= (1200,730))

    
    window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
    window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
    window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
    window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

    while True:
        event, values = window.read()
        #print("")
        #print(event)
        #print(values)
        if event == sg.WIN_CLOSED:
            break
## --------------- PLUS MINUS BUTTONS --------------------------- ##
        elif event == ('minus'):
            image_box_size = image_box_size-5
            window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
        elif event == ('plus'):
            image_box_size = image_box_size+5
            window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

## --------------- STUDY SELECT RADIO BUTTONS ------------------- ##
        elif event == ('mepoRadio'):
            IRB = 'Mepo'
            window['genxeInputs'].update(visible=False)
            window['mepoInputs'].update(visible=True)
            window['clinicalInputs'].update(visible=False)
        elif event == ('genxeRadio'):
            IRB = 'GenXe'
            window['genxeInputs'].update(visible=True)
            window['mepoInputs'].update(visible=False)
            window['clinicalInputs'].update(visible=False)
        elif event == ('clinicalRadio'):
            IRB = 'Clinical'
            window['genxeInputs'].update(visible=False)
            window['mepoInputs'].update(visible=False)
            window['clinicalInputs'].update(visible=True)
        #elif event == ('clinicalRadio'):
        #    window['clinicalInputs'].update(visible=True)


## --------------- INITIALIZE Button ------------------- ##
        elif event == ('-INITIALIZE-'):
            DICOM_path = values['path1']
            MASK_path = values['path2']
            TWIX_path = values['path3']
            window['-CALCVDP-'].update(button_color = 'lightgray')
            window['-CALCCI-'].update(button_color = 'lightgray')
            window['-RUNTWIX-'].update(button_color = 'lightgray')
            window['-EXPORT-'].update(button_color = 'lightgray')
            try:
                del Vent1
                window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
                window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
                window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
                window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
                window['subject'].update(f'Subject: ')
                window['studydate'].update(f'Study Date: ')
                window['studytime'].update(f'Study Time: ')
                window['age'].update(f'Age: ')
                window['sex'].update(f'Sex: ')
                window['dob'].update(f'DOB: ')
                window['height'].update(f'Height: ')
                window['weight'].update(f'Weight: ')
                window['vox'].update(f'')
                window['snr'].update(f'SNR:')
                window['vdp'].update(f'VDP: ')
                window['ventarrayshape'].update(f'Ventilation Array Shape: ')
                window['masklungvol'].update(f'Mask Lung Volume:')
                window['defectvolume'].update(f'Defect Volume:')
                window['ci'].update(f'CI:')
                window['twixdate'].update(f'Twix Date:')
                window['twixprotocol'].update(f'Twix Protocol:')
                print('cleared Vent1')
            except:
                print('cache already clean')
            try:
                Vent1 = Vent_Analysis(DICOM_path,MASK_path)
                window['-STATUS-'].update("Vent_Analysis loaded",text_color='green')
                window['-INITIALIZE-'].update(button_color = 'green')
                window['subject'].update(f'Subject: {Vent1.PatientName}')
                window['studydate'].update(f'Study Date: {Vent1.StudyDate}')
                window['studytime'].update(f'Study Time: {Vent1.StudyTime}')
                window['age'].update(f'Age: {Vent1.PatientAge}')
                window['sex'].update(f'Sex: {Vent1.PatientSex}')
                window['dob'].update(f'DOB: {Vent1.PatientBirthDate}')
                window['height'].update(f'Height: {Vent1.PatientSize} [m]')
                window['weight'].update(f'Weight: {Vent1.PatientWeight} [kg]')
                window['vox'].update(f'DICOM voxel Size: {Vent1.vox} [mm]')
                rawMontage = Vent1.array3D_to_montage2D(Vent1.HPvent)
                rawMontageImage = arrayToImage(255*normalize(rawMontage),(int(image_box_size*rawMontage.shape[1]/rawMontage.shape[0]),image_box_size))
                window['-RAWIMAGE-'].update(data=rawMontageImage)
            except:
                window['-STATUS-'].update("ERROR: Uhh you messed something up. Maybe check your DICOM and MASK paths?",text_color='red')
                continue

## --------------- CALCULATE VDP Button ------------------- ##
        elif event == ('-CALCVDP-'):
            try:
                window['-STATUS-'].update("Calculating VDP...",text_color='blue')
                Vent1.runVDP()
                window['snr'].update(f'SNR: {np.round(Vent1.SNR,1)}')
                window['vdp'].update(f'VDP: {np.round(Vent1.VDP,1)} [%]')
                window['ventarrayshape'].update(f'Ventilation Array Shape: {np.round(Vent1.HPvent.shape)}')
                window['masklungvol'].update(f'Mask Lung Volume: {np.round(np.sum(Vent1.mask>0)*np.prod(Vent1.vox)/1000/1000,1)} [L]')
                window['defectvolume'].update(f'Defect Volume: {np.round(np.sum(Vent1.defectArray>0)*np.prod(Vent1.vox)/1000/1000,1)} [L]')
                window['-STATUS-'].update("VDP Calculated",text_color='green')
                window['-CALCVDP-'].update(button_color = 'green')
                N4Montage = Vent1.array3D_to_montage2D(Vent1.N4HPvent)
                mask_border = Vent1.array3D_to_montage2D(Vent1.mask_border)
                N4MontageImage = arrayToImage(colorBinary(N4Montage,mask_border),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-N4IMAGE-'].update(data=N4MontageImage)
                DefectMontage = Vent1.array3D_to_montage2D(Vent1.defectArray)
                DefectMontageImage = arrayToImage(colorBinary(N4Montage,DefectMontage),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-DEFECTIMAGE-'].update(data=DefectMontageImage)
            except:
                window['-STATUS-'].update("ERROR: VDP couldnt run for some reason...",text_color='red')
                continue

## --------------- CALCULATE CI Button ------------------- ##
        elif event == ('-CALCCI-'):
            try:
                window['-STATUS-'].update("Calculating CI...",text_color='blue')
                Vent1.calculate_CI()
                window['ci'].update(f'CI: {np.round(Vent1.CI,1)}')
                window['-STATUS-'].update("CI Calculated successfully",text_color='green')
                window['-CALCCI-'].update(button_color = 'green')
            except:
                window['-STATUS-'].update("ERROR: CI couldnt run for some reason...",text_color='red')
                continue

## --------------- RUN TWIX Button ------------------- ##
        elif event == ('-RUNTWIX-'):
            try:
                TWIX_path = values['path3']
                window['-STATUS-'].update("Processing TWIX file...",text_color='blue')
                Vent1.process_RAW(TWIX_path)
                window['-STATUS-'].update("TWIX Processed successfully",text_color='green')
                window['-RUNTWIX-'].update(button_color = 'green')
                TwixMontage = Vent1.array3D_to_montage2D(Vent1.raw_HPvent)
                TwixMontageImage = arrayToImage(255*normalize(TwixMontage),(int(image_box_size*TwixMontage.shape[1]/TwixMontage.shape[0]),image_box_size))
                window['-TWIXIMAGE-'].update(data=TwixMontageImage)
                window['twixdate'].update(f'Twix Date: {Vent1.scanDateTime}')
                window['twixprotocol'].update(f'Twix Protocol: {Vent1.protocolName}')
            except:
                window['-STATUS-'].update("ERROR: TWIX couldnt process for some reason...",text_color='red')
                continue

## --------------- EXPORT Button ------------------- ##
        elif event == ('-EXPORT-'):
            window['-STATUS-'].update("Exporting Data...",text_color='blue')
            EXPORT_path = values['exportpath']
            errorMarker = []

            # Did the user input their name??
            if values['userName'] == '':
                window['-STATUS-'].update("Don't forget to enter your Name or Initials at the very top right!...",text_color='red')
                continue

            # Create the fileName from Input Data
            treatment = 'none'
            visit = '0'
            if values['genxeRadio']:
                fileName = f"Xe-{values['genxeID']}_{Vent1.StudyDate[2:]}"
                if values['prealb']: fileName = f'{fileName}_preAlb';treatment = 'preAlb'
                elif values['postalb']: fileName = f'{fileName}_postAlb';treatment = 'postAlb'
                elif values['presil']: fileName = f'{fileName}_preSil';treatment = 'preSilo'
                elif values['postsil']: fileName = f'{fileName}_postSil';treatment = 'postSil'
            elif values['mepoRadio']:
                fileName = f"Mepo{values['mepoID']}_{Vent1.StudyDate[2:]}"
                if values['mepoVisit1']: fileName = f'{fileName}_visit1';visit = 1
                elif values['mepoVisit2']: fileName = f'{fileName}_visit2';visit = 2
                elif values['mepoVisit3']: fileName = f'{fileName}_visit3';visit = 3
                if values['prealb_mepo']: fileName = f'{fileName}_preAlb';treatment = 'preAlb'
                elif values['postalb_mepo']: fileName = f'{fileName}_postAlb';treatment = 'postAlb'
            elif values['clinicalRadio']:
                fileName = f"Clinical_{values['clinicalID']}_{Vent1.StudyDate[2:]}_visit{values['clinicalvisitnumber']}"
                if values['singlesession']: fileName = f'{fileName}_singlesession';treatment = 'none'
                elif values['prealb_clin']: fileName = f'{fileName}_preAlb';treatment = 'preAlb'
                elif values['postalb_clin']: fileName = f'{fileName}_postAlb';treatment = 'postAlb'
                
            print(f'-- FileName: {fileName} --')

            # Create the Export directory
            if not os.path.isdir(EXPORT_path):
                os.makedirs(EXPORT_path)

            hf = h5py.File(os.path.join(EXPORT_path,fileName),'w')
            archive_flag = False
            if values['-ARCHIVE-'] == True:
                print('Archiving')
                archive_flag = os.path.isdir(ARCHIVE_path)
                if(not archive_flag): 
                    print('\033[33mArchive path does not exist, and so data will not be archived\033[37m')
                else:
                    ha = h5py.File(os.path.join(ARCHIVE_path,fileName),'w')
            

            # 1 - build the METADATA into the h5
            try:
                print('\033[34mBuilding Metadata into h5..\033[37m')
                metadata = Vent1.buildMetadata()
                metadata['visit'] = visit
                metadata['treatment'] = treatment
                metadata['fileName'] = fileName
                metadata['DE'] = values['DE']
                metadata['FEV1'] = values['FEV1']
                metadata['FVC'] = values['FVC']
                metadata['IRB'] = IRB
                metadata['SNR'] = str(Vent1.SNR)
                metadata['VDP'] = str(Vent1.VDP)
                metadata['LungVolume'] = str(np.round(np.sum(Vent1.mask>0)*np.prod(Vent1.vox)/1000/1000,1))
                metadata['DefectVolume'] = str(np.round(np.sum(Vent1.defectArray>0)*np.prod(Vent1.vox)/1000/1000,1))
            except:
                print('\033[31mError at building metadata...\033[37m')
            try:
                metadata['CI'] = str(Vent1.CI)
            except:
                print('\033[33mNoCI to add...\033[37m')
            try:
                hf.attrs.update(metadata)
                if archive_flag: ha.attrs.update(metadata)
            except:
                print('\033[31mError at building metadata...\033[37m')
                errorMarker = np.append(errorMarker,1)

            # 2 - build the 4D data arrays and add to h5
            try:
                print('\033[34mBuilding 4D dataArrays and adding to h5...\033[37m')
                hf.create_dataset('dataArray',data = Vent1.build4DdataArray())
                if archive_flag: ha.create_dataset('dataArray',data = Vent1.build4DdataArray())
            except:
                errorMarker = np.append(errorMarker,2)
                print('\033[31mError building 4D arrays...\033[37m')

            # 3 - Export the 4D dataArray as a Nifti
            try:
                print('\033[34mExporting 4D arrays as Niftis...\033[37m')
                Vent1.exportNifti(EXPORT_path,fileName)
            except:
                errorMarker = np.append(errorMarker,3)
                print('\033[31mError exporing 4D arrays as Nifits...\033[37m')

            # 4 - build and add the twixArray to H5
            try:
                print('\033[34mExporting TWIX arrays as Niftis...\033[37m')
                twixArray = np.zeros((Vent1.raw_HPvent.shape[0],Vent1.raw_HPvent.shape[1],Vent1.raw_HPvent.shape[2],2)).astype(np.complex128)
                twixArray[:,:,:,0] = Vent1.raw_HPvent
                twixArray[:,:,:,1] = np.transpose(Vent1.raw_K,(1,0,2))
                hf.create_dataset('twixArrayReal',data = np.real(twixArray))
                hf.create_dataset('twixArrayImag',data = np.imag(twixArray))
                if archive_flag: ha.create_dataset('twixArrayReal',data = np.real(twixArray)) 
                if archive_flag: ha.create_dataset('twixArrayImag',data = np.imag(twixArray)) 
            except:
                errorMarker = np.append(errorMarker,4)
                print('\033[31mError exporting TWIX as Niftis...\033[37m')

            # 5 - Save TWIX Header to JSON
            try:
                twix_header = Vent1.raw_twix
                del twix_header['image']
                with open(os.path.join(EXPORT_path,f'{fileName}_TWIXHEADER.json'), 'w') as fp:
                    json.dump(extract_attributes(twix_header), fp,indent=2)
            except:
                errorMarker = np.append(errorMarker,5)
                print('\033[31mError saving JSON header...\033[37m')

            # Report which errors occured to the GUI
            if len(errorMarker) == 0:
                window['-STATUS-'].update("Data Exported successfully!",text_color='green')
                window['-EXPORT-'].update(button_color = 'green')
            else:
                window['-STATUS-'].update(f"Error markers {errorMarker} were flagged",text_color='red')

            hf.close()
            if values['-ARCHIVE-'] == True:
                ha = hf
                ha.close()
                print('Archived')

hf.close()



