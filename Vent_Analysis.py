## -- PIRL Ventilation Image Analysis Pipeline -- ##
## -- GMGD, 7/1/2024 -- ##
import CI # ------------------------------- for calculateCI
import json # ----------------------------- For saving header as json file
import nibabel as nib # ------------------- for Nifti stuffs
import numpy as np
import os
import pickle # --------------------------- For Pickling and unpickling data
from PIL import Image, ImageTk, ImageDraw, ImageFont # ---------- for arrayToImage conversion
import pydicom as dicom # ----------------- for openSingleDICOM and openDICOMFolder
from scipy.signal import medfilt2d # ------ for calculateVDP
import SimpleITK as sitk # ---------------- for N4 Bias Correection
import skimage.util # --------------------- for image montages
from sys import getsizeof # --------------- To report twix object size
import time # ----------------------------- for calculateVDP
import tkinter as tk # -------------------- GUI stuffs
from tkinter import filedialog # ---------- for openSingleDICOM and openDICOMFolder
import mapvbvd # -------------------------- for process_Raw
from sklearn.cluster import KMeans # ---------- for kMeans VDP
#from matplotlib import pyplot as plt # ---- for makeSlide and screenShot
import matplotlib.pyplot as plt


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
    ATTRIBUTES: (all are declared in __init__ with empty strings, most are self explanatory but heres a list of a few)
        version - Date and author of most recent Vent_Analysis update
        N4HPvent - N4 bias corrected ventilation array
        defectArray - binary array of defect voxels (using mean-anchored array)
        VDP - the ventilation defect percentage using 60% treshold by default (using mean-anchored array)
        CIarray - the Cluster Index Array given the defectArray and vox
        CI - the 95th percentile Cluster Value
        vox - the dimensions of a voxel as a vector
        ds - the complete pydicom object for the HPvent
        metadata - a dictionary of single value properties of the data
    METHODS:
        __init__ - Opens the HP Vent and mask dicoms into self.HPvent, and self.mask
        openSingleDICOM - opens a single 3D dicom file (for vent or proton images)
        openDICOMfolder - opens all 2D dicoms in a folder (for mask images) into 3D array
        pullDICOMHeader - pulls useful DICOM header info into self variables
        calculate_VDP - creates N4HPvent, defectarrays and VDPs
        calculate_CI - calculates CI (imports CI functions)
        exportDICOM - creates a DICOM of the ventilation with defect overlay
        cropToData - using the mask, returns only the rows,cols,slices with signal
        screenShot - Produces a png image of the data
        process_RAW - process the corresponding TWIX file associated
        pickleMe - creates a dictionary of all class attributes and saves as pickle
        unPickleMe - given a pickle, rebuilds the class object
    """
    def __init__(self,xenon_path = None, 
                 mask_path = None, 
                 proton_path = None,
                 xenon_array = None,
                 mask_array=None,
                 proton_array=None,
                 pickle_dict = None,
                 pickle_path = None):
        
        self.version = '240701_GMGD' # - update this when changes are made!! - #
        self.proton = ''
        self.N4HPvent = ''
        self.defectArray = ''
        self.CIarray = ''
        self.vox = ''
        self.ds = ''
        self.twix = ''
        self.raw_k = ''
        self.raw_HPvent = ''
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
                        'LungVolume': '',
                        'DefectVolume': '',
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
        '''Pulls some of our favorite elements from the DICOM header and adds to metadata dicionary. Voxel dimensions 'vox' are stored separately'''
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

        self.metadata['LungVolume'] = np.sum(Vent1.mask == 1)*np.prod(Vent1.vox)/1000000

    def calculateBorder(self,A):
        '''Given a binary array, returns the border of the binary volume (useful for creating a border from the mask for display)'''
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
        '''Given HPvent and a mask calculates VDPs'''
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

        self.metadata['DefectVolume'] = np.sum(Vent1.defectArray == 1)*np.prod(Vent1.vox)/1000000

        ## -- K-Means [Kirby, 2012] -- ##
        # k = KMeans(signal_list)
        # k.fit(np.array(signal_list).reshape(-1,1))
        
        print('\033[32mcalculate_VDP ran successfully\033[37m')

    def calculate_CI(self):
        '''Calculates the Cluster Index Array and reports the subject's cluster index (CI)'''
        self.CIarray = CI.calculate_CI(self.defectArray,self.vox)
        CVlist = np.sort(self.CIarray[self.defectArray>0])
        index95 = int(0.95*len(CVlist))
        self.metadata['CI'] = CVlist[index95]
        print(f"Calculated CI: {self.metadata['CI']}")

    def exportNifti(self,filepath=None,fileName = None):
        '''Builds all arrays into a 4D array (see build4DdataArrays()) and exports as Nifti'''
        print('\033[34mexportNifti method called...\033[37m')
        if filepath == None:
            print('\033[94mWhere would you like to save your Niftis?\033[37m')
            filepath = tk.filedialog.askdirectory()

        if fileName == None:
            fileName = str(self.metadata['PatientName']).replace('^','_')

        try:
            dataArray = self.build4DdataArray()
            niImage = nib.Nifti1Image(dataArray, affine=np.eye(4))
            savepath = os.path.join(filepath,fileName + '_dataArray.nii')
            nib.save(niImage,savepath)
            print(f'\033[32mNifti HPvent array saved to {savepath}\033[37m')
        except:
            print('\033[31mCould not Export 4D HPvent mask Nifti...\033[37m')

    def build4DdataArray(self):
        ''' Our arrays are: Proton [0], HPvent [1], mask  [2], N4HPvent [3], defectArray [4], CIarray [5]'''
        dataArray = np.zeros((self.HPvent.shape[0],self.HPvent.shape[1],self.HPvent.shape[2],6),dtype=np.float32)
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
        '''Performs N4itk Bias Correction'''
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
        '''Given a pydicom object (elem) extracts all elements and builds into dictionary'''
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
        '''Saves DICOM Header to json file (except the Pixel Data)'''
        dicom_dict = self.dicom_to_dict(ds, include_private)
        with open(json_path, 'w') as json_file:
            json.dump(dicom_dict, json_file, indent=4)
        print(f"\033[32mJson file saved to {json_path}\033[37m")

    def exportDICOM(self,ds,save_dir = 'C:/PIRL/data/',optional_text = '',forPACS=True):
        '''Create and saves the Ventilation images with defectArray overlayed
        Note: Our PACS doesn't seem to like the export of 3D data as a single DICOM...'''
        if self.metadata['VDP'] == '':
            print('\033[31mCant export dicoms until you run calculate_VDP()...\033[37m')
        else:
            BW = (self.normalize(np.abs(self.N4HPvent)) * (2 ** 8 - 1)).astype('uint%d' % 8)
            RGB = np.zeros((self.N4HPvent.shape[0],self.N4HPvent.shape[1],self.N4HPvent.shape[2],3),dtype=np.uint8)
            RGB[:,:,:,0] = BW*(self.defectArray==0) + 255*(self.defectArray==1)
            RGB[:,:,:,1] = BW*(self.defectArray==0)
            RGB[:,:,:,2] = BW*(self.defectArray==0)
            if forPACS is False:
                RGB = np.transpose(RGB,axes = (2,0,1,3)) # first dimension must always be slices for DICOM export
                ds.PhotometricInterpretation = 'RGB'
                ds.SamplesPerPixel = 3
                ds.Rows, ds.Columns, ds.NumberOfFrames  = BW.shape
                ds.BitsAllocated = ds.BitsStored = 8
                ds.HighBit = 7
                ds.SOPInstanceUID = ds.SeriesInstanceUID = dicom.uid.generate_uid()
                ds.PixelData = RGB.tobytes()
                ds.SeriesDescription = f"{optional_text} - VDP: {np.round(self.metadata['VDP'],1)}"
                save_path = os.path.join(save_dir,f"{self.metadata['PatientName']}_defectDICOM.dcm")
                ds.save_as(save_path)
                print(f'\033[32mdefect DICOM saved to {save_path}\033[37m')
            elif forPACS is True:
                new_uid = dicom.uid.generate_uid()
                self.ds.SeriesInstanceUID = new_uid
                num_dicoms = BW.shape[2]
                all_locations = range(0, num_dicoms, 1)
                num_bits = 8
                dicom_path = os.path.join(save_dir,'defectDICOMS')
                if not os.path.isdir(dicom_path):
                    os.makedirs(dicom_path)
                for i in range(num_dicoms):
                    color_image = RGB[:,:,i,:]
                    ds.PixelData = color_image.tobytes()
                    ds.Rows, ds.Columns = color_image.shape[:2]
                    ds.SamplesPerPixel = 3
                    ds.PhotometricInterpretation = 'RGB'
                    ds.BitsAllocated = num_bits
                    ds.BitsStored = num_bits
                    ds.SeriesDescription = f"{optional_text} - VDP: {np.round(self.metadata['VDP'],1)}"
                    ds.InstanceNumber = i + 1
                    ds.SliceLocation = all_locations[i]
                    new_uid = dicom.uid.generate_uid()
                    ds.SOPInstanceUID = new_uid
                    ds.NumberOfFrames = 1
                    ds.save_as(os.path.join(dicom_path, f"dicom_{i}.dcm"))
    
    def array3D_to_montage2D(self,A):
        return skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], grid_shape = (1,A.shape[2]), padding_width=0, fill=0)

    def cropToData(self, A, border=0,borderSlices=False):
        '''Given a 3D mask array, crops rows,cols,slices to only those with signal (useful for creating montage slides)'''
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
        '''Creates and saves a montage image of all processed data images'''
        from matplotlib.colors import LinearSegmentedColormap

        cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
        [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
        [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
        0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
        [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
        0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
        [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
        0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
        [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
        0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
        [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
        0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
        [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
        0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
        0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
        [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
        0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
        [0.0589714286, 0.6837571429, 0.7253857143], 
        [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
        [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
        0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
        [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
        0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
        [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
        0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
        [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
        0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
        [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
        [0.7184095238, 0.7411333333, 0.3904761905], 
        [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
        0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
        [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
        [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
        0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
        [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
        0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
        [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
        [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
        [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
        0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
        [0.9763, 0.9831, 0.0538]]
        
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
        A[:,:,:,5] = self.normalize(A[:,:,:,5]) # -- CI
        mask_border = self.mask_border[np.ix_(rr,cc,ss)]
        rr = A.shape[0]
        cc = A.shape[1]
        ss = A.shape[2]
        imageArray = np.zeros((rr*5,cc*ss,3))
        for s in range(ss):
            # -- Blank (for header info)
            imageArray[0:rr,(0+s*cc):(cc + s*cc),0] = A[:,:,s,0]*0
            imageArray[0:rr,(0+s*cc):(cc + s*cc),1] = A[:,:,s,0]*0
            imageArray[0:rr,(0+s*cc):(cc + s*cc),2] = A[:,:,s,0]*0

            # -- proton
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,0]
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,0]
            imageArray[(rr):(2*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,0]

            # -- raw xenon
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,1]
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,1]
            imageArray[(rr*2):(3*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,1]
            
            # -- N4 xenon w mask border
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,3]*(1-mask_border[:,:,s])
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,3]*(1-mask_border[:,:,s]) + mask_border[:,:,s]
            imageArray[(rr*3):(4*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,3]*(1-mask_border[:,:,s]) + mask_border[:,:,s]

            # -- N4 xenon w defects
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),0] = A[:,:,s,3]*(1-A[:,:,s,4]) + A[:,:,s,4]
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),1] = A[:,:,s,3]*(1-A[:,:,s,4])
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),2] = A[:,:,s,3]*(1-A[:,:,s,4])

            # -- CI
            # Apply Parula colormap
            import matplotlib.pyplot as plt
            parula_cmap = LinearSegmentedColormap.from_list('parula', cm_data)
            # Apply colormap to A[:,:,s,5] and assign to imageArray
            colored_image = parula_cmap(A[:,:,s,5])
            imageArray[(rr*4):(5*rr),(0+s*cc):(cc + s*cc),:] = colored_image[..., :3]

            # Display or use imageArray as needed
            #plt.imshow(imageArray)
            #plt.colorbar()
            #plt.show()

        #plt.imsave(path, imageArray) # -- matplotlib command to save array as png
        image = Image.fromarray(np.uint8(imageArray*255))  # Convert the numpy array to a PIL image
        draw = ImageDraw.Draw(image)
        text_info = ['','','','']
        text_info[0] = f"Patient: {self.metadata['PatientName']} -- {self.metadata['PatientAge']}-{self.metadata['PatientSex']}"
        text_info[1] = f"StudyDate: {self.metadata['StudyDate']} -- {self.metadata['visit']}/{self.metadata['treatment']}"
        #text_info[2] = f"FEV1: {self.metadata['FEV1']} -- VDP: {np.round(self.metadata['VDP'],1)}"
        text_info[2] = f"FEV1: {self.metadata['FEV1']} -- VDP: {np.round(self.metadata['VDP'], 1)} -- CI: {self.metadata['CI']}"

        for k in range(len(text_info)):
            draw.text((10,10+k*40),text_info[k],fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 40))
        draw.text((np.round(imageArray.shape[1]*.75),10),f'Analysis Version: {self.version}',fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 40))
        image.save(path, 'PNG')  # Save the image
        print(f'\033[32mScreenshot saved to {path}\033[37m')

    def process_RAW(self,filepath=None):
        '''Given a twix file, will extract the raw kSpace and reconstruct the image array.
            3 new attributes are created here from the twix file: the twix object, the raw kSpace array, and the image array (kSpace fft'd).
            I'm not a huge fan of this because having objects as class attributes isn't super backwards compatible, nor convertable across versions.
            This is why when we pickle the Vent_Analysis object, all attributes are converted to dictionaries first (I learned that the hard way).
            Anyways, in the future I think we need to do the same here: extract twix object attributes to a dictionary. But idk. 
            The ISMRMRD can help here, BUT header info is lost in this conversion so I don't want to do it exclusively...'''
        if filepath == None:
            print('\033[94mSelect the corresponding RAW data file (Siemens twix)...\033[37m\n')
            filepath = tk.filedialog.askopenfilename()
        self.raw_twix = mapvbvd.mapVBVD(filepath)
        self.metadata['TWIXscanDateTime'] = self.raw_twix.hdr.Config['PrepareTimestamp']
        self.metadata['TWIXprotocolName'] = self.raw_twix.hdr.Meas['tProtocolName']
        self.raw_twix.image.squeeze = True
        self.raw_K = self.raw_twix.image['']
        self.raw_HPvent = np.zeros((self.raw_K.shape)).astype(np.complex128)
        for k in range(self.raw_K.shape[2]):
            self.raw_HPvent[:,:,k] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.raw_K[:,:,k])))
        self.raw_HPvent = np.transpose(self.raw_HPvent,(1,0,2))[:,::-1,:]

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


# - CI test code
# vox = [3.22,1.98,15]
# mg = np.meshgrid(np.arange(0,100,vox[0]),np.arange(0,100,vox[1]),np.arange(0,100,vox[2]))
# defectArray = ((50-mg[0])**2 + (50-mg[1])**2 + (50-mg[2])**2) < 20**2
# plt.imshow(defectArray[:,int(defectArray.shape[1]/2),:]);plt.show()
# CIarray = CI.calculate_CI(defectArray,vox = vox)
# np.max(CIarray)
# 20*2**(1/3)

# # # #Some test code
# VDPs = []
# CIs = []
# fileNames = []
# parent = '//umh.edu/data/Radiology/Xenon_Studies/Gaby/240425_CI/240405_VDP_analysis/Pkls'
# all_paths = os.listdir(parent)
# for pickle_file in all_paths:
#     with open(os.path.join(parent,pickle_file), 'rb') as file:
#         print(os.path.join(parent,pickle_file))
#         data = pickle.load(file)
#     Vent1 = Vent_Analysis(xenon_array=data[0][:,:,:,1],mask_array=data[0][:,:,:,2])
#     Vent1.proton = data[0][:,:,:,0]
#     if len(data) == 2:
#         Vent1.metadata = data[1]
#         vox = data[1]['DICOMVoxelSize']
#     elif len(data) == 3:
#         Vent1.metadata = data[2]
#         Vent1.raw_K = data[1][:,:,:,0]
#         Vent1.raw_HPvent = data[1][:,:,:,1]
#         vox = data[2]['DICOMVoxelSize']
#     float_list = vox.strip('[]').split(', ')
#     float_list = [float(x) for x in float_list]
#     vox = np.array(float_list)
#     Vent1.vox = vox
#     Vent1.calculate_VDP()
#     Vent1.calculate_CI()
#     VDPs = np.append(VDPs,Vent1.metadata['VDP'])
#     CIs = np.append(CIs,Vent1.metadata['CI'])
#     fileNames = np.append(fileNames,pickle_file)


# signal_list = sorted(Vent1.N4HPvent[Vent1.mask>0])
# k = KMeans(n_clusters=5)
# k.fit(np.array(signal_list).reshape(-1,1))

# # Vent1.calculate_VDP()
# CVlist = np.sort(CIarray[defectArray>0])
# index95 = int(0.95*len(CVlist))
# metadata['CI'] = CVlist[index95]
# print(f"Calculated CI: {metadata['CI']}")

# # #Some test code
# pickle_path = 'C:/Users/rptho/Downloads/Mepo0006_211213_visit1_preAlb.pkl'
# with open(pickle_path, 'rb') as file:
#     data = pickle.load(file)
# Vent1 = Vent_Analysis(xenon_array=data[0][:,:,:,1],mask_array=data[0][:,:,:,2])
# Vent1.proton = data[0][:,:,:,0]
# Vent1.metadata = data[2]
# Vent1.raw_K = data[1][:,:,:,0]
# Vent1.raw_HPvent = data[1][:,:,:,1]
# vox = data[2]['DICOMVoxelSize']vox
# float_list = vox.strip('[]').split(', ')
# float_list = [float(x) for x in float_list]
# vox = np.array(float_list)
# Vent1.vox = vox
# Vent1
# Vent1.calculate_VDP()
# Vent1.calculate_CI()

# Vent1.calculate_VDP()
# Vent1.metadata['FEV1'] = 95
# Vent1.screenShot()
# Vent1.dicom_to_json(Vent1.ds)
# Vent1.exportDICOM(Vent1.ds,save_dir='C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_240509/',forPACS=True)
# Vent1.exportDICOM(Vent1.ds,save_dir='C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_240509/',forPACS=False)

# Vent1.metadata['VDP']
# Vent1.metadata['VDP_lb']
# Vent1.pickleMe(pickle_path = f"c:/PIRL/data/{Vent1.metadata['PatientName']}.pkl")
# Vent2 = Vent_Analysis(pickle_path=f"c:/PIRL/data/{Vent1.metadata['PatientName']}.pkl")



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

if __name__ == "__main__":
    version = '240516_RPT'
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
                           [sg.Radio('Baseline','clinicalalbuterol',key='baseline'),
                            sg.Radio('Albuterol','clinicalalbuterol',key='albuterol')],
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
            window['masklungvol'].update(f"Mask Lung Volume: {str(Vent1.metadata['LungVolume'])} [L]")
            try:
                window['defectvolume'].update(f"Defect Volume: {str(Vent1.metadata['DefectVolume'])} [L]")
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
                if values['baseline']: fileName = f'{fileName}_baseline';Vent1.metadata['treatment'] = 'none'
                elif values['albuterol']: fileName = f'{fileName}_Albuterol';Vent1.metadata['treatment'] = 'Albuterol'
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
            Vent1.exportDICOM(Vent1.ds,EXPORT_path,optional_text=fileName,forPACS=True)
            window['-STATUS-'].update("Data Successfully Exported...",text_color='green')

            if values['-ARCHIVE-'] == True:
                if os.path.isdir(ARCHIVE_path):
                    Vent1.pickleMe(pickle_path=os.path.join(ARCHIVE_path,f'{fileName}.pkl'))
                    window['-STATUS-'].update("Data Successfully Exported and Archived...",text_color='green')
                else:
                    window['-STATUS-'].update("Data Successfully Exported but not Archived...",text_color='orange')
                    print("Cant Archive because the path doesn't exist...")


            


'''Things to add (updated 3/27/2024):
 - CI colormap output in screenshot and GUI
 - Multiple VDPs calculated (linear binning, k-means) (LB done)
 - show histogram?
 - edit mask
 - automatic segmentation using proton (maybe DL this?)
 - Denoise Option
 '''



