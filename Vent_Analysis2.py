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
#from matplotlib import pyplot as plt # ---- for makeSlide and screenShot
import skimage.util # --------------------- for image montages
import nibabel as nib # ------------------- for Nifti stuffs
import PySimpleGUI as sg # ---------------- for GUI stuffs
from PIL import Image, ImageTk # ---------- for arrayToImage conversion
import pickle # --------------------------- For Pickling and unpickling data
import json # ----------------------------- For saving header as json file

#------------------------------------------------------------------------------------
# ----------- VENTILATION ANALYSIS CLASS DEFINITION ---------------------------------
#------------------------------------------------------------------------------------
class Vent_Analysis:
    """Performs complete VDP analysis: N4Bias correction, normalization,
        defect calculation, and VDP calculation.
    INPUTS: 
    2 inputs are required at minimum:
        HPvent - 3D array of ventilation image stack
        mask - 3D array of lung segmentation for HPvent (must match HPvent shape)
    these inputs can be called either by direct input as numpy arrays, as paths to dicom files/folders, or as a pickle file to be unpickled
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
        openSingleDICOM - opens a single 3D dicom file (for vent or proton images)
        openDICOMfolder - opens all 2D dicoms in a folder (for mask images) into 3D array
        pullDICOMHeader - pulls useful DICOM header info into self variables
        runVDP - creates N4HPvent, defectarray (mean-anchored) and VDP
        calculateCI - calculates CI (imports CI functions)
        process_RAW - process the corresponding TWIX file associated
    """
    def __init__(self,xenon_path = None, 
                 mask_path = None, 
                 proton_path = None,
                 xenon_array = None,
                 mask_array=None,
                 proton_array=None,
                 pickle_dict = None,
                 pickle_path = None):
        
        self.version = '240504_RPT' # - update this when changes are made!! - #
        self.proton = ''
        self.N4HPvent = ''
        self.defectArray = ''
        self.CIarray = ''
        self.vox = ''
        self.ds = ''
        self.twix = ''
        # self.raw_k = ''
        # self.raw_HPvent = ''
        self.metadata = {'fileName': '',
                        'PatientName': '',
                        'PatientAge': '',
                        'PatientBirthDate' : '',
                        'PatientSex': '',
                        'StudyDate': '',
                        'SeriesTime': '',
                        'DE': '',
                        'SNR': '',
                        'VDP': '',
                        'VDP_lb': '',
                        'VDP_km': '',
                        'CI': '',
                        'FEV1': '', 
                        'FVC': '',
                        'visit': '',
                        'IRB': '',
                        'treatment': '',
                        'notes': ''
                        # 'TWIXprotocolName': '',
                        # 'TWIXscanDateTime': ''
                        }


        ## -- Was the xenon array provided or a path to its DICOM? -- ##
        if xenon_array is not None:
            print(f'\033[34mXenon array provided: {xenon_array.shape}\033[37m')
            self.HPvent = xenon_array

        if xenon_path is not None:
            try:
                print('\033[34mXenon DICOM path provided. Opening DICOM...\033[37m')
                self.ds, self.HPvent = self.openSingleDICOM(xenon_path)
            except:
                print('\033[31mOpening Xenon DICOM failed...\033[37m')

            try:
                print('\033[34mPulling Xenon DICOM Header\033[37m')
                self.pullDICOMHeader()
            except:
                print('\033[31mPulling Xenon DICOM Header failed...\033[37m')

        ## -- Was the mask array provided or a path to its DICOM folder? -- ##
        if mask_array is not None:
            print(f'\033[34mMask array provided: {mask_array.shape}\033[37m')
            self.mask = mask_array
            self.mask_border = self.calculateBorder(self.mask)

        if mask_path is not None:
            try:
                print('\033[34mLoading Mask and calculating border\033[37m')
                _, self.mask = self.openDICOMfolder(mask_path)
                self.mask_border = self.calculateBorder(self.mask)
            except:
                print('\033[31mLoading Mask and calculating border failed...\033[37m')


        ## -- Was a proton array provided or a path to its DICOM? -- ##
        if proton_array is not None: 
            print(f'\033[34mProton array provided: {proton_array.shape}\033[37m')
            self.proton = proton_array

        if proton_path is not None:
            if proton_path is not None: 
                try:
                    print('\033[34mProton DICOM Path provided. Opening...\033[37m')
                    self.proton_ds, self.proton = self.openSingleDICOM(proton_path)
                except:
                    print('\033[31mOpening Proton DICOM failed...\033[37m')


        ## -- Was a pickle or a pickle path provided? -- ##
        if pickle_path is not None:
            print(f'\033[34mPickle path provided: {pickle_path}. Loading...\033[37m')
            try:
                with open(pickle_path, 'rb') as file:
                    pickle_dict = pickle.load(file)
                print(f'\033[32mPickle file successfully loaded.\033[37m')
            except:
                print('\033[31mOpening Pickle from path and building arrays failed...\033[37m')

        if pickle_dict is not None:
            self.unPickleMe(pickle_dict)

        
    def openSingleDICOM(self,dicom_path):        
        if dicom_path is None:
            root = tk.Tk()
            root.withdraw()
            print('\033[94mSelect the DICOM ventilation file...\033[37m')
            dicom_path = tk.filedialog.askopenfilename()
            ds = dicom.dcmread(dicom_path,force=True)
        else:
            ds = dicom.dcmread(dicom_path)
        DICOM_array = ds.pixel_array
        DICOM_array = np.transpose(DICOM_array,(1,2,0))
        print(f'\033[32mI opened a DICOM of shape {DICOM_array.shape}\033[37m')
        return ds, DICOM_array


    def openDICOMfolder(self,maskFolder):  
        from tkinter import filedialog
        if maskFolder is None:
            print('\033[94mSelect the mask folder...\033[37m')
            maskFolder = tk.filedialog.askdirectory()
        dcm_filelist = [f for f in sorted(os.listdir(maskFolder)) if f.endswith('.dcm')]
        ds = dicom.dcmread(os.path.join(maskFolder,dcm_filelist[0]))
        mask = np.zeros((ds.pixel_array.shape[0],ds.pixel_array.shape[1],len(dcm_filelist)))
        for f,k in zip(dcm_filelist,range(len(dcm_filelist))):
            ds = dicom.dcmread(os.path.join(maskFolder,f))
            mask[:,:,k] = ds.pixel_array
        print(f'\033[32mI built a mask of shape {mask.shape}\033[37m')
        return ds, mask

    def pullDICOMHeader(self):
        infoList = ['PatientName','PatientAge','PatientBirthDate','PatientSize','PatientWeight','PatientSex','StudyDate','StudyTime','SeriesTime']
        for elem in infoList:
            try:
                self.metadata[elem] = self.ds[elem].value
            except:
                print(f'\033[31mNo {elem}\033[37m')
                self.metadata[elem] = ''

        for k in range(100):
            try:
                self.vox = self.ds[0x5200, 0x9230][k]['PixelMeasuresSequence'][0].PixelSpacing
                break
            except:
                if k == 99:
                    print('Pixel Spacing not in correct place in DICOM header, please enter each dimension...')
                    self.vox = [float(input()),float(input())]

        try:
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(self.ds.SpacingBetweenSlices)]
        except:
            print('Slice spacing not in correct position in DICOM header. Please enter manually:')
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(input())]

    def calculateBorder(self,A):
        border = np.zeros(A.shape)
        for k in range(A.shape[2]):
            x = np.gradient(A[:,:,k].astype(float))
            border[:,:,k] = (x[0]!=0)+(x[1]!=0)
        return border

    def normalize(self,x):
        if (np.max(x) - np.min(x)) == 0:
            return x
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    def calculate_VDP(self,thresh=0.6):
        self.metadata['SNR'] = self.calculate_SNR(self.HPvent,self.mask) ## -- SNR of xenon DICOM images, not Raw, nor N4
        self.N4HPvent = self.N4_bias_correction(self.HPvent,self.mask)

        ## -- Mean-anchored Linear Binning [Thomen et al. 2015 Radiology] -- ##
        signal_list = sorted(self.N4HPvent[self.mask>0])
        mean_normalized_vent = np.divide(self.N4HPvent,np.mean(signal_list))
        self.defectArray = np.zeros(mean_normalized_vent.shape)
        for k in range(self.mask.shape[2]):
            self.defectArray[:,:,k] = medfilt2d((mean_normalized_vent[:,:,k]<thresh)*self.mask[:,:,k])
        self.defectBorder = self.calculateBorder(self.defectArray) == 1
        self.metadata['VDP'] = 100*np.sum(self.defectArray)/np.sum(self.mask)

        ## -- Linear Binning [Mu He, 2016] -- ##
        norm95th_vent = np.divide(self.N4HPvent,signal_list[int(len(signal_list)*.99)])
        self.defectArrayLB = ((norm95th_vent<=0.16)*1 + (norm95th_vent>0.16)*(norm95th_vent<=0.34)*2 + (norm95th_vent>0.34)*(norm95th_vent<=0.52)*3 + (norm95th_vent>0.52)*(norm95th_vent<=0.7)*4 + (norm95th_vent>0.7)*(norm95th_vent<=0.88)*5 + (norm95th_vent>0.88)*6)*self.mask
        self.metadata['VDP_lb'] = 100*np.sum((self.defectArrayLB == 1)*1 + (self.defectArrayLB == 2)*1)/np.sum(self.mask)

        ## -- K-Means [Kirby, 2012] -- ##
        print('\033[32mcalculate_VDP ran successfully\033[37m')

    def calculate_CI(self):
        '''Calculates the Cluster Index Array and reports the subject's cluster index (CI)'''
        self.CIarray = CI.calculate_CI(self.defectArray,self.vox)
        CVlist = np.sort(self.CIarray[self.defectArray>0])
        index95 = int(0.95*len(CVlist))
        self.metadata['CI'] = CVlist[index95]
        print(f"Calculated CI: {self.metadata['CI']}")
        #return self.CIarray, self.CI

    def exportNifti(self,filepath=None,fileName = None):
        print('\033[34mexportNifti method called...\033[37m')
        if filepath == None:
            print('\033[94mWhere would you like to save your Niftis?\033[37m')
            filepath = tk.filedialog.askdirectory()

        if fileName == None:
            fileName = str(self.metadata['PatientName']).replace('^','_')

        try:
            dataArray = self.build4DdataArray()
            niImage = nib.Nifti1Image(dataArray, affine=np.eye(4))
            #niImage.header['pixdim'] = self.vox
            savepath = os.path.join(filepath,fileName + '_dataArray.nii')
            nib.save(niImage,savepath)
            print(f'\033[32mNifti HPvent array saved to {savepath}\033[37m')
        except:
            print('\033[31mCould not Export 4D HPvent mask Nifti...\033[37m')

    def build4DdataArray(self):
        ''' Our arrays are: Proton [0], HPvent [1], mask  [2], N4HPvent [3], defectArray [4], CIarray [5]'''
        dataArray = np.zeros((self.HPvent.shape[0],self.HPvent.shape[1],self.HPvent.shape[2],6))
        dataArray[:,:,:,1] = self.HPvent
        dataArray[:,:,:,2] = self.mask
        try:
            dataArray[:,:,:,0] = self.proton
        except:
            print('\033[33mProton either does not exist or does not match Xenon array shape and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,3] = self.N4HPvent
        except:
            print('\033[33mN4HPvent does not exist and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,4] = self.defectArray
        except:
            print('\033[33mdefectArray does not exist and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,5] = self.CIarray
        except:
            print('\033[33mCIarray does not exist and was not added to 4D array\033[37m')
        return dataArray


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
            pass
            #sub_array = hpg.get_subarray(self.HPvent[:,:,int(self.HPvent.shape[2]/2)])
            #noise = sub_array['A'].flatten()
        SNR = (np.mean(signal)-np.mean(noise))/np.std(noise)
        return SNR
    

    def dicom_to_dict(self, elem, include_private=False):
        data_dict = {}
        for sub_elem in elem:
            if not include_private and sub_elem.tag.is_private:
                continue
            if sub_elem.name in ['Pixel Data']:
                continue
            if sub_elem.VR == "SQ":  # Sequence of items
                data_dict[sub_elem.name] = [self.dicom_to_dict(item, include_private) for item in sub_elem.value]
            else:
                data_dict[sub_elem.name] = str(sub_elem.value)
        return data_dict

    def dicom_to_json(self, ds, json_path='c:/pirl/data/DICOMjson.json', include_private=True):
        dicom_dict = self.dicom_to_dict(ds, include_private)
        with open(json_path, 'w') as json_file:
            json.dump(dicom_dict, json_file, indent=4)
        print(f"\033[32mJson file saved to {json_path}\033[37m")
    
    def array3D_to_montage2D(self,A):
        return skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], grid_shape = (1,A.shape[2]), padding_width=0, fill=0)

    def cropToData(self, A, border=0,borderSlices=False):
        # Calculate the indices for non-zero slices, rows, and columns
        slices = np.multiply(np.sum(np.sum(A,axis=0),axis=0)>0,list(range(0,A.shape[2])))
        rows = np.multiply(np.sum(np.sum(A,axis=1),axis=1)>0,list(range(0,A.shape[0])))
        cols = np.multiply(np.sum(np.sum(A,axis=2),axis=0)>0,list(range(0,A.shape[1])))
        
        # Filter out the indices for non-zero slices, rows, and columns
        slices = [x for x in range(0,A.shape[2]) if slices[x]]
        rows = [x for x in range(0,A.shape[0]) if rows[x]]
        cols = [x for x in range(0,A.shape[1]) if cols[x]]
        
        # Add border, ensuring we don't exceed the array's original dimensions
        if borderSlices:
            slices_start = max(slices[0] - border, 0)
            slices_end = min(slices[-1] + border + 1, A.shape[2])
        else:
            slices_start = max(slices[0] , 0)
            slices_end = min(slices[-1] + 1, A.shape[2])
        rows_start = max(rows[0] - border, 0)
        rows_end = min(rows[-1] + border + 1, A.shape[0])
        cols_start = max(cols[0] - border, 0)
        cols_end = min(cols[-1] + border + 1, A.shape[1])
        
        # Crop the array with the adjusted indices
        cropped_A = A[rows_start:rows_end, cols_start:cols_end, slices_start:slices_end]
        return cropped_A, list(range(rows_start, rows_end)), list(range(cols_start, cols_end)), list(range(slices_start, slices_end))

    def screenShot(self, path = 'C:/PIRL/data/screenShotTest.png', normalize95 = False):
        A = self.build4DdataArray()
        _,rr,cc,ss = self.cropToData(A[:,:,:,2],border = 5)
        A = A[np.ix_(rr,cc,ss,np.arange(A.shape[3]))]
        A[:,:,:,0] = self.normalize(A[:,:,:,0]) # -- proton
        if normalize95:
            signalList = A[:,:,:,1].flatten()
            signalList.sort()
            A[:,:,:,1] = np.divide(A[:,:,:,1],signalList[int(len(signalList)*0.99)])
            A[:,:,:,1][A[:,:,:,1]>1] = 1
        A[:,:,:,1] = self.normalize(A[:,:,:,1]) # -- raw xenon
        A[:,:,:,2] = self.normalize(A[:,:,:,2]) # -- mask
        A[:,:,:,3] = self.normalize(A[:,:,:,3]) # -- N4 xenon
        A[:,:,:,4] = self.normalize(A[:,:,:,4]) # -- defectArray
        mask_border = self.mask_border[np.ix_(rr,cc,ss)]
        rr = A.shape[0]
        cc = A.shape[1]
        ss = A.shape[2]
        imageArray = np.zeros((rr*5,cc*ss,3))
        for s in range(ss):
            # -- proton
            imageArray[0:rr,(0+s*cc):(cc + s*cc),0] = A[:,:,s,0]
            imageArray[0:rr,(0+s*cc):(cc + s*cc),1] = A[:,:,s,0]
            imageArray[0:rr,(0+s*cc):(cc + s*cc),2] = A[:,:,s,0]

            # -- raw xenon
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,1]
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,1]
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,1]

            # -- N4 xenon
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,3]
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,3]
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,3]
            
            # -- N4 xenon w mask border
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,3]*(1-mask_border[:,:,s])
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,3]*(1-mask_border[:,:,s]) + mask_border[:,:,s]
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,3]*(1-mask_border[:,:,s]) + mask_border[:,:,s]

            # -- N4 xenon w defects
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,3]*(1-A[:,:,s,4]) + A[:,:,s,4]
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,3]*(1-A[:,:,s,4])
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,3]*(1-A[:,:,s,4])
        #plt.imsave(path, imageArray) # -- matplotlib command to save array as png
        image = Image.fromarray(np.uint8(imageArray*255))  # Convert the numpy array to a PIL image
        image.save(path, 'PNG')  # Save the image
        print(f'\033[32mScreenshot saved to {path}\033[37m')



    def pickleMe(self, pickle_path):
        '''Uses dictionary comprehension to create a dictionary of all class attriutes, then saves as pickle'''
        pickle_dict = {attr: getattr(self, attr) for attr in vars(self)}
        with open(pickle_path, 'wb') as file:
            pickle.dump(pickle_dict, file)
        print(f'\033[32mPickled dictionary saved to {pickle_path}\033[37m')


    def unPickleMe(self,pickle_dict):
        '''Given a pickled dictionary (yep, I actually named a variable pickle_dict), it will extract entries to class attributes'''
        for attr, value in pickle_dict.items():
            setattr(self, attr, value)


    def __repr__(self):
        string = (f'\033[35mVent_Analysis\033[37m class object version \033[94m{self.version}\033[37m\n')
        for attr, value in vars(self).items():
            if value == '':
                string += (f'\033[31m {attr}: \033[37m\n')
            elif type(value) is np.ndarray:
                string += (f'\033[32m {attr}: \033[36m{value.shape} \033[37m\n')
            elif type(value) is dict:
                for attr2, value2 in value.items():
                    if value2 == '':
                        string += (f'   \033[31m {attr2}: \033[37m\n')
                    else:
                        string += (f'   \033[32m {attr2}: \033[36m{value2} \033[37m\n')
            else:
                string += (f'\033[32m {attr}: \033[36m{type(value)} \033[37m\n')
        return string

# #Some test code
# Vent1 = Vent_Analysis(xenon_path='C:/PIRL/data/MEPOXE0039/48522586xe',mask_path='C:/PIRL/data/MEPOXE0039/Mask')
# Vent1
# Vent1.calculate_VDP()
# Vent1.screenShot()
# Vent1.dicom_to_json(Vent1.ds)
# Vent1.metadata['VDP']
# Vent1.metadata['VDP_lb']
# Vent1.pickleMe(pickle_path = f"c:/PIRL/data/{Vent1.metadata['PatientName']}.pkl")
# Vent2 = Vent_Analysis(pickle_path=f"c:/PIRL/data/{Vent1.metadata['PatientName']}.pkl")



    # def extractPickle(self,pkl,version):
    #     self.proton = pkl[0][:,:,:,0]
    #     self.HPvent = pkl[0][:,:,:,1]
    #     self.mask = pkl[0][:,:,:,2]
    #     self.N4HPvent = pkl[0][:,:,:,3]
    #     self.defectArray = pkl[0][:,:,:,4]
    #     self.CIarray = pkl[0][:,:,:,5]
    #     self.mask_border = self.calculateBorder(self.mask)
    #     try:
    #         self.version = pkl[1]['version'];print(f'Name: {self.version}')
    #         self.PatientName = pkl[1]['DICOMPatientName'];print(f'Name: {self.PatientName}')
    #         self.StudyDate = pkl[1]['DICOMStudyDate'];print(f'Study Date: {self.StudyDate}')
    #         self.StudyTime = pkl[1]['DICOMStudyTime']
    #         self.PatientAge = pkl[1]['DICOMPatientAge']
    #         self.PatientBirthDate = pkl[1]['DICOMPatientBirthDate'];print(f'Patient BirthDate: {self.PatientBirthDate}')
    #         self.PatientSex = pkl[1]['DICOMPatientSex']
    #         self.PatientSize = pkl[1]['DICOMPatientHeight']
    #         self.PatientWeight = pkl[1]['DICOMPatientWeight']
    #         self.vox = [float(pkl[1]['DICOMVoxelSize'][1:4]),
    #                     float(pkl[1]['DICOMVoxelSize'][6:9]),
    #                     float(pkl[1]['DICOMVoxelSize'][11:15])]
    #         print(f'DICOMVoxelSize: {self.vox}')
    #         print('\033[32mMetadata pull from Pickle was Successful...\033[37m')
    #     except:
    #         print('\033[31mMetadata pull from Pickle Failed with pkl[1]...\033[37m')

    #     try:
    #         self.version = pkl[1]['version'];print(f'Name: {self.version}')
    #         self.PatientName = pkl[1]['DICOMPatientName'];print(f'Name: {self.PatientName}')
    #         self.StudyDate = pkl[1]['DICOMStudyDate'];print(f'Study Date: {self.StudyDate}')
    #         self.StudyTime = pkl[1]['DICOMStudyTime']
    #         self.PatientAge = pkl[1]['DICOMPatientAge']
    #         self.PatientBirthDate = pkl[1]['DICOMPatientBirthDate'];print(f'Patient BirthDate: {self.PatientBirthDate}')
    #         self.PatientSex = pkl[1]['DICOMPatientSex']
    #         self.PatientSize = pkl[1]['DICOMPatientHeight']
    #         self.PatientWeight = pkl[1]['DICOMPatientWeight']
    #         self.vox = [float(pkl[1]['DICOMVoxelSize'][1:4]),
    #                     float(pkl[1]['DICOMVoxelSize'][6:9]),
    #                     float(pkl[1]['DICOMVoxelSize'][11:15])]
    #         print(f'DICOMVoxelSize: {self.vox}')
    #         print('\033[32mMetadata pull from Pickle Failed...\033[37m')
    #     except:
    #         print('\033[31mMetadata pull from Pickle Failed...\033[37m')
        

    # def process_RAW(self,filepath=None):
    #     if filepath == None:
    #         print('\033[94mSelect the corresponding RAW data file (Siemens twix)...\033[37m\n')
    #         filepath = tk.filedialog.askopenfilename()
    #     self.raw_twix = mapvbvd.mapVBVD(filepath)
    #     self.metadata['TWIXscanDateTime'] = self.raw_twix.hdr.Config['PrepareTimestamp']
    #     self.metadata['TWIXprotocolName'] = self.raw_twix.hdr.Meas['tProtocolName']
    #     self.raw_twix.image.squeeze = True
    #     self.raw_K = self.raw_twix.image['']
    #     self.raw_HPvent = np.zeros((self.raw_K.shape)).astype(np.complex128)
    #     for k in range(self.raw_K.shape[2]):
    #         self.raw_HPvent[:,:,k] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.raw_K[:,:,k])))
    #     self.raw_HPvent = np.transpose(self.raw_HPvent,(1,0,2))[:,::-1,:]

# def extract_attributes(attr_dict, parent_key='', sep='_'):
#     """
#     Recursively extract all attributes and subattributes from a nested dictionary and compiles into flat dictionary.
    
#     Args:
#     - attr_dict (dict): The attribute dictionary to extract from.
#     - parent_key (str): The base key to use for building key names for subattributes.
#     - sep (str): The separator to use between nested keys.
    
#     Returns:
#     - dict: A flat dictionary with all attributes and subattributes.
#     """
#     items = []
#     for k, v in attr_dict.items():
#         new_key = f"{parent_key}{sep}{k}" if parent_key else k
#         if isinstance(v, dict):
#             # If the value is a dictionary, recurse
#             items.extend(extract_attributes(v, new_key, sep=sep).items())
#         else:
#             # Otherwise, add the attribute to the items list
#             items.append((new_key, v))
#     return dict(items)


### ------------------------------------------------------------------------------------------------ ###
### ---------------------------------------- MAIN SCRIPT ------------------------------------------- ###
### ------------------------------------------------------------------------------------------------ ###

if __name__ == "__main__":
    version = '240505_RPT'
    image_box_size = 50
    ARCHIVE_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/Archive/'
    
    import PySimpleGUI as sg
    from datetime import date # -- So we can export the analysis date

    ## -- Helper Functions for GUI -- ##
    def arrayToImage(A,size):
        imgAr = Image.fromarray(A.astype(np.uint8))
        imgAr = imgAr.resize(size)
        image = ImageTk.PhotoImage(image=imgAr)
        return(image)

    def normalize(x):
        if (np.max(x) - np.min(x)) == 0:
            return x
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    def colorBinary(A,B):
        A = normalize(A)
        new = np.zeros((A.shape[0],A.shape[1],3))
        new[:,:,0] = A*(B==0) + B
        new[:,:,1] = A*(B==0)
        new[:,:,2] = A*(B==0)
        return new*255
    
    sg.theme('Default1')
    PIRLlogo = os.path.join(os.getcwd(),'PIRLlogo.png')
    path_label_column = [[sg.Text('Path to Ventilation DICOM:')],[sg.Text('Path to Mask Folder:')],[sg.Text('Path to Proton:')],[sg.Text('Path to Twix:')]]
    path_column = [[sg.InputText(key='DICOMpath',default_text='C:/PIRL/data/MEPOXE0039/48522586xe',size=(200,200))],
                   [sg.InputText(key='MASKpath',default_text='C:/PIRL/data/MEPOXE0039/Mask',size=(200,200))],
                   [sg.InputText(key='PROTONpath',default_text='C:/PIRL/data/MEPOXE0039/48522597prot',size=(200,200))],
                   [sg.InputText(key='TWIXpath',default_text='C:/PIRL/data/MEPOXE0039/meas_MID00077_FID58046_6_FLASH_gre_hpg_2201_SliceThi_10.dat',size=(200,200))]]
    
    IRB_select_column = [
                    [sg.Radio('GenXe','IRB',key='genxeRadio',enable_events=True)],
                    [sg.Radio('Mepo','IRB',key='mepoRadio',enable_events=True)],
                    [sg.Radio('Clinical','IRB',key='clinicalRadio',enable_events=True)]]
    genxe_info_column = [[sg.Text('General Xenon ID:'),sg.InputText(default_text='0000',size=(10,10),key='genxeID')],
                           [sg.Text('Disease:'),sg.Radio('Healthy','disease',key='diseaseHealthy'),sg.Radio('Asthma','disease',key='diseaseAsthma'),sg.Radio('CF','disease',key='diseaseCF'),sg.Radio('COPD','disease',key='diseaseCOPD'),sg.Radio('Other:','disease',key='diseaseOther'),sg.InputText(size=(10,1))],
                           [sg.Checkbox('PreAlbuterol',default=False,key='prealb'),sg.Checkbox('PostAlbuterol',default=False,key='postalb'),sg.Checkbox('PreSildenafil',default=False,key='presil'),sg.Checkbox('PostSildenafil',default=False,key='postsil')],
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

    patient_data_column = [[sg.Button('',key='editPatientName',pad=(0,0)),sg.Text('Subject:                               ',key='subject',pad=(0,0))],
                           [sg.Button('',key='editStudyDate',pad=(0,0)),sg.Text('Study Date:',key='studydate',pad=(0,0))],
                           [sg.Button('',key='editStudyTime',pad=(0,0)),sg.Text('Study Time:',key='studytime',pad=(0,0))],
                           [sg.Button('',key='editTwixDate',pad=(0,0)),sg.Text('Twix Date:',key='twixdate',pad=(0,0))],
                           [sg.Button('',key='editProtocol',pad=(0,0)),sg.Text('Protocol:',key='twixprotocol',pad=(0,0))],
                           [sg.Button('',key='editPatientAge',pad=(0,0)),sg.Text('Age:',key='age',pad=(0,0))],
                           [sg.Button('',key='editPatientSex',pad=(0,0)),sg.Text('Sex:',key='sex',pad=(0,0))],
                           [sg.Button('',key='editPatientDOB',pad=(0,0)),sg.Text('DOB:',key='dob',pad=(0,0))],]
    dicom_data_column = [[sg.Text('DICOM Voxel Size:                                ',key = 'vox',pad=(0,0))],
                         [sg.Text('SNR:',key = 'snr',pad=(0,0))],
                         [sg.Text('VDP:',key = 'vdp',pad=(0,0))],
                         [sg.Text('Ventilation Array Shape:',key='ventarrayshape',pad=(0,0))],
                         [sg.Text('Mask Lung Vol:',key='masklungvol',pad=(0,0))],
                         [sg.Text('Defect Volume:',key='defectvolume',pad=(0,0))],
                         [sg.Text('CI:',key='ci',pad=(0,0))]]
    image_column = [[sg.Image(key='-PROTONIMAGE-')],
                    [sg.Image(key='-RAWIMAGE-')],
                    [sg.Image(key='-N4IMAGE-')],
                    [sg.Image(key='-DEFECTIMAGE-')],
                    [sg.Image(key='-TWIXIMAGE-')]]

    layout = [
        [sg.Image(PIRLlogo),sg.Text(f'version {version}'),sg.Text('         User:'),sg.InputText(key='userName',size=(10,1),enable_events=False),sg.Button('-',key='minus'),sg.Button('+',key='plus')],
        [sg.HorizontalSeparator()],
        [sg.Column(path_label_column),sg.Column(path_column)],
        [sg.Button('Load from Paths', key='-INITIALIZE-'),sg.Button('Calculate VDP', key='-CALCVDP-'),sg.Button('Calculate CI', key='-CALCCI-'),sg.Button('Import TWIX', key='-RUNTWIX-'),sg.Button('Load Pickle', key='-LOADPICKLE-',pad = (300,0))],  
        [sg.HorizontalSeparator()],
        [sg.Column(IRB_select_column),
         sg.Column(clinical_info_column,key='clinicalInputs',visible=False),
         sg.Column(genxe_info_column,key='genxeInputs',visible=False),
         sg.Column(mepo_info_column,key='mepoInputs',visible=False),
         sg.Column(dose_info_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(patient_data_column),sg.VSeperator(),sg.Column(dicom_data_column),sg.VSeperator(),sg.Column(image_column)],  
        [sg.Text('Notes:'),sg.InputText(key='notes',size=(200,200))],
        [sg.Text('',key = '-STATUS-')],
        [sg.Text('Export Path:'),sg.InputText(key='exportpath',default_text='C:/PIRL/data/MEPOXE0039/',size=(200,200))],
        [sg.Button('Export Data',key='-EXPORT-'),sg.Checkbox('Copy pickle to Archive',default=True,key='-ARCHIVE-'),sg.Push(),sg.Button('Clear Cache',key='-CLEARCACHE-')]
    ]

    window = sg.Window(f'PIRL Ventilation Analysis -- {version}', layout, return_keyboard_events=True, margins=(0, 0), finalize=True, size= (1200,730))

    def updateImages():
            window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            try:
                protonMontage = Vent1.array3D_to_montage2D(Vent1.proton)
                protonMontageImage = arrayToImage(255*normalize(protonMontage),(int(image_box_size*protonMontage.shape[1]/protonMontage.shape[0]),image_box_size))
                window['-PROTONIMAGE-'].update(data=protonMontageImage)
            except:
                window['-PROTONIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                rawMontage = Vent1.array3D_to_montage2D(Vent1.HPvent)
                mask_border = Vent1.array3D_to_montage2D(Vent1.mask_border)
                rawMontageImage = arrayToImage(colorBinary(rawMontage,mask_border),(int(image_box_size*rawMontage.shape[1]/rawMontage.shape[0]),image_box_size))
                window['-RAWIMAGE-'].update(data=rawMontageImage)
            except:
                window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                N4Montage = Vent1.array3D_to_montage2D(Vent1.N4HPvent)
                mask_border = Vent1.array3D_to_montage2D(Vent1.mask_border)
                N4MontageImage = arrayToImage(colorBinary(N4Montage,mask_border),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-N4IMAGE-'].update(data=N4MontageImage)
            except:
                window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                DefectMontage = Vent1.array3D_to_montage2D(Vent1.defectArray)
                DefectMontageImage = arrayToImage(colorBinary(N4Montage,DefectMontage),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-DEFECTIMAGE-'].update(data=DefectMontageImage)
            except:
                window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

    def updateData():
        if 'Vent1' in globals():
            window['subject'].update(f"Subject: {Vent1.metadata['PatientName']}")
            window['studydate'].update(f"Study Date: {Vent1.metadata['StudyDate']}")
            window['studytime'].update(f"Study Time: {Vent1.metadata['StudyTime']}")
            window['age'].update(f"Age: {Vent1.metadata['PatientAge']}")
            window['sex'].update(f"Sex: {Vent1.metadata['PatientSex']}")
            window['dob'].update(f"DOB: {Vent1.metadata['PatientBirthDate']}")
            window['vox'].update(f"DICOM voxel Size: {Vent1.vox} [mm]")
            window['snr'].update(f"SNR: {Vent1.metadata['SNR']}")
            window['vdp'].update(f"VDP: {Vent1.metadata['VDP']}")
            window['ventarrayshape'].update(f'Ventilation Array Shape: {Vent1.HPvent.shape}')
            window['masklungvol'].update(f'Mask Lung Volume: {str(np.sum(Vent1.mask == 1)*np.prod(Vent1.vox)/1000000)} [L]')
            try:
                window['defectvolume'].update(f'Defect Volume: {str(np.sum(Vent1.defectArray == 1)*np.prod(Vent1.vox)/1000000)} [L]')
                window['ci'].update(f"CI: {Vent1.metadata['CI']}")
            except:
                pass
            window['twixdate'].update(f'Twix Date:')
            window['twixprotocol'].update(f'Twix Protocol:')

    updateImages()
    updateData()

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
            updateImages()
        elif event == ('plus'):
            image_box_size = image_box_size+5
            updateImages()

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

## --------------- Info Edit Buttons ------------------- ##
        elif event == ('editPatientName'):
            text = sg.popup_get_text('Enter Subject ID: ')
            window['subject'].update(f'Subject: {text}')
            Vent1.metadata['PatientName'] = text
        elif event == ('editPatientAge'):
            text = sg.popup_get_text('Enter Patient Age: ')
            window['age'].update(f'Age: {text}')
            Vent1.metadata['PatientAge'] = text
        elif event == ('editPatientSex'):
            text = sg.popup_get_text('Enter Patient Sex: ')
            window['sex'].update(f'Sex: {text}')
            Vent1.metadata['PatientSex'] = text
        elif event == ('editPatientDOB'):
            text = sg.popup_get_text('Enter Patient DOB: ')
            window['dob'].update(f'DOB: {text}')
            Vent1.metadata['PatientDOB'] = text

## --------------- Load Pickle ------------------- ##       
        elif event == ('-LOADPICKLE-'):
            pickle_path = sg.popup_get_text('Enter Pickle Path: ',default_text='//umh.edu/data/Radiology/Xenon_Studies/Studies/Archive/Mepo0029_231030_visit1_preAlb.pkl')
            Vent1 = Vent_Analysis(pickle_path=pickle_path)
            window['-STATUS-'].update("Vent_Analysis pickle loaded",text_color='green')
            window['-INITIALIZE-'].update(button_color = 'green')
            updateData()
            updateImages()

## --------------- INITIALIZE Button ------------------- ##
        elif event == ('-INITIALIZE-'):
            DICOM_path = values['DICOMpath']
            MASK_path = values['MASKpath']
            TWIX_path = values['TWIXpath']
            PROTON_path = values['PROTONpath']
            window['-CALCVDP-'].update(button_color = 'lightgray')
            window['-CALCCI-'].update(button_color = 'lightgray')
            window['-RUNTWIX-'].update(button_color = 'lightgray')
            window['-EXPORT-'].update(button_color = 'lightgray')
            try:
                del Vent1
                print('cleared Vent1')
            except:
                print('cache already clean')
            try:
                if PROTON_path == '':
                    Vent1 = Vent_Analysis(DICOM_path,MASK_path)
                else:
                    Vent1 = Vent_Analysis(DICOM_path,MASK_path,PROTON_path)
                window['-STATUS-'].update("Vent_Analysis loaded",text_color='green')
                window['-INITIALIZE-'].update(button_color = 'green')
                updateData()
                updateImages()
            except:
                window['-STATUS-'].update("ERROR: Uhh you messed something up. Maybe check your DICOM and MASK paths?",text_color='red')
                continue

## --------------- CALCULATE VDP Button ------------------- ##
        elif event == ('-CALCVDP-'):
            try:
                window['-STATUS-'].update("Calculating VDP...",text_color='blue')
                Vent1.calculate_VDP()
                window['-STATUS-'].update("VDP Calculated",text_color='green')
                window['-CALCVDP-'].update(button_color = 'green')
                updateImages()
                updateData()
            except:
                window['-STATUS-'].update("ERROR: VDP either couldnt run or be displayed for some reason...",text_color='red')
                continue

## --------------- CALCULATE CI Button ------------------- ##
        elif event == ('-CALCCI-'):
            try:
                window['-STATUS-'].update("Calculating CI...",text_color='blue')
                Vent1.calculate_CI()
                window['-STATUS-'].update("CI Calculated successfully",text_color='green')
                window['-CALCCI-'].update(button_color = 'green')
                updateImages()
                updateData()
            except:
                window['-STATUS-'].update("ERROR: CI couldnt run for some reason...",text_color='red')
                continue

## --------------- RUN TWIX Button ------------------- ##
        elif event == ('-RUNTWIX-'):
            pass
            # try:
            #     TWIX_path = values['TWIXpath']
            #     window['-STATUS-'].update("Processing TWIX file...",text_color='blue')
            #     Vent1.process_RAW(TWIX_path)
            #     window['-STATUS-'].update("TWIX Processed successfully",text_color='green')
            #     window['-RUNTWIX-'].update(button_color = 'green')
            #     TwixMontage = Vent1.array3D_to_montage2D(Vent1.raw_HPvent)
            #     TwixMontageImage = arrayToImage(255*normalize(TwixMontage),(int(image_box_size*TwixMontage.shape[1]/TwixMontage.shape[0]),image_box_size))
            #     window['-TWIXIMAGE-'].update(data=TwixMontageImage)
            #     window['twixdate'].update(f'Twix Date: {Vent1.scanDateTime}')
            #     window['twixprotocol'].update(f'Twix Protocol: {Vent1.protocolName}')
            # except:
            #     window['-STATUS-'].update("ERROR: TWIX couldnt process for some reason...",text_color='red')
            #     continue

## --------------- CLEAR CACHE Button ------------------- ##
        elif event == ('-CLEARCACHE-'):
            try:
                del Vent1
            except:
                pass
            print('Clearing Cache...')
            window['notes'].update('')
            window['-INITIALIZE-'].update(button_color = 'lightgray')
            window['-CALCVDP-'].update(button_color = 'lightgray')
            window['-CALCCI-'].update(button_color = 'lightgray')
            window['-RUNTWIX-'].update(button_color = 'lightgray')
            window['-EXPORT-'].update(button_color = 'lightgray')
            window['genxeRadio'].update(False)
            window['genxeInputs'].update(visible=False)
            window['mepoRadio'].update(False)
            window['mepoInputs'].update(visible=False)
            window['clinicalRadio'].update(False)
            window['clinicalInputs'].update(visible=False)
            updateData()
            updateImages()
            window['-STATUS-'].update("Analysis Cache is cleared and ready for the next subject!...",text_color='blue')

            

## --------------- EXPORT Button ------------------- ##
        elif event == ('-EXPORT-'):
            '''Here we'll save 3 important things: the 4D data arrays, one of the DICOM datasets and one of the TWIX datasets,
            and all the many single variable inputs/outputs such as patient name, study ID, scan date/time, etc. To do this, we'll
            pickle everything in a single all-in-one file to be saved in the specified path and if desired in a static 'archive path, 
            and separately we'll save the arrays as Nifti's and full headers for TWIX as JSON files.'''

            # Did the user input their name??
            if values['userName'] == '':
                window['-STATUS-'].update("Don't forget to enter your Name or Initials at the very top right!...",text_color='red')
                continue

            # Did the user select an IRB??
            if not values['genxeRadio'] and not values['mepoRadio'] and not values['clinicalRadio']:
                window['-STATUS-'].update("Don't forget to select an IRB!...",text_color='red')
                continue

            # Create the EXPORT_path and fileName and populate class metadata dictionary with values from GUI input fields
            window['-STATUS-'].update("Exporting Data...",text_color='blue')
            today = date.today().strftime("%y%m%d")
            user = values['userName']
            targetPath = f'VentAnalysis_{user}_{today}/'
            EXPORT_path = os.path.join(values['exportpath'],targetPath)
            treatment = 'none'
            visit = '0'
            if values['genxeRadio']:
                fileName = f"Xe-{values['genxeID']}_{Vent1.metadata['StudyDate'][2:]}"
                if values['prealb']: fileName = f'{fileName}_preAlb';Vent1.metadata['treatment'] = 'preAlbuterol'
                elif values['postalb']: fileName = f'{fileName}_postAlb';Vent1.metadata['treatment'] = 'postAlbuterol'
                elif values['presil']: fileName = f'{fileName}_preSil';Vent1.metadata['treatment'] = 'preSildenafil'
                elif values['postsil']: fileName = f'{fileName}_postSil';Vent1.metadata['treatment'] = 'postSildenafil'
            elif values['mepoRadio']:
                fileName = f"Mepo{values['mepoID']}_{Vent1.metadata['StudyDate'][2:]}"
                if values['mepoVisit1']: fileName = f'{fileName}_visit1';Vent1.metadata['visit'] = 1
                elif values['mepoVisit2']: fileName = f'{fileName}_visit2';Vent1.metadata['visit'] = 2
                elif values['mepoVisit3']: fileName = f'{fileName}_visit3';Vent1.metadata['visit'] = 3
                if values['prealb_mepo']: fileName = f'{fileName}_preAlb';Vent1.metadata['treatment'] = 'preAlb'
                elif values['postalb_mepo']: fileName = f'{fileName}_postAlb';Vent1.metadata['treatment'] = 'postAlb'
            elif values['clinicalRadio']:
                fileName = f"Clinical_{values['clinicalID']}_{Vent1.metadata['StudyDate'][2:]}_visit{values['clinicalvisitnumber']}"
                if values['singlesession']: fileName = f'{fileName}_singlesession';Vent1.metadata['treatment'] = 'none'
                elif values['prealb_clin']: fileName = f'{fileName}_preAlb';Vent1.metadata['treatment'] = 'preAlbuterol'
                elif values['postalb_clin']: fileName = f'{fileName}_postAlb';Vent1.metadata['treatment'] = 'postAlbuterol'
            print(f'-- FileName: {fileName} --')
            print(f'-- FilePath: {EXPORT_path} --')
            if not os.path.isdir(EXPORT_path):
                os.makedirs(EXPORT_path)
            try:
                Vent1.metadata['fileName'] = fileName
                Vent1.metadata['DE'] = values['DE']
                Vent1.metadata['FEV1'] = values['FEV1']
                Vent1.metadata['FVC'] = values['FVC']
                Vent1.metadata['IRB'] = IRB
                Vent1.metadata['notes'] = values['notes']
            except:
                window['-STATUS-'].update("Could not add GUI metadata values to Class metadata...",text_color='red')
                print('\033[31mError adding GUI data to class metadata...\033[37m')

            #Export Nifti Arrays, DICOM header json, Class pickle, and screenshot
            Vent1.exportNifti(EXPORT_path,fileName)
            Vent1.dicom_to_json(Vent1.ds, json_path=os.path.join(EXPORT_path,f'{fileName}.json'))
            Vent1.pickleMe(pickle_path=os.path.join(EXPORT_path,f'{fileName}.pkl'))
            Vent1.screenShot(path=os.path.join(EXPORT_path,f'{fileName}.png'))
            window['-STATUS-'].update("Data Successfully Exported...",text_color='green')

            if values['-ARCHIVE-'] == True:
                if os.path.isdir(ARCHIVE_path):
                    Vent1.pickleMe(pickle_path=os.path.join(ARCHIVE_path,f'{fileName}.pkl'))
                    window['-STATUS-'].update("Data Successfully Exported and Archived...",text_color='green')
                else:
                    window['-STATUS-'].update("Data Successfully Exported but not Archived...",text_color='orange')
                    print("Cant Archive because the path doesn't exist...")


            


'''Things to add (updated 3/27/2024):
 - Vent_Analysis class inputs either paths to data or the arrays themselves (done 5/4/2024)
 - Output DICOM header info as JSON (done)
 - CI colormap output in screenshot
 - Multiple VDPs calculated (linear binning, k-means) (LB done)
 - show histogram?
 - edit mask
 - automatic segmentation using proton (maybe DL this?)
 - Denoise Option
 '''



