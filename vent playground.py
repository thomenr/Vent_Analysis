import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('c:/vent_analysis/')
from Vent_Analysis import Vent_Analysis
import skimage.util # --------------------- for image montages

Vent2 = Vent_Analysis(pickle_path="C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_241006/Mepo0039_240301.pkl")
Vent2.metadata['analysisUser'] = 'RPT'
Vent2.metadata['Disease'] = 'asthma'
Vent2.screenShot()


def normalize(x):
    if (np.max(x) - np.min(x)) == 0:
        return x
    else:
        return (x - np.min(x)) / (np.max(x) - np.min(x))

def makeMontage(A,nRows = None,nCols = None,sameScale = False):
    ''' given a 3D array returns a 2D montage'''
    if nRows is not None:
        nCols = int(np.ceil(A.shape[2]/nRows))
    if nCols is not None:
        nRows = int(np.ceil(A.shape[2]/nCols))
    if nRows is None and nCols is None:
        nRows = nCols = int(np.ceil(np.sqrt(A.shape[2])))
    k=0
    for R in range(nRows):
        for C in range(nCols):
            if k == A.shape[2]:
                D = np.zeros((A.shape[0],CC.shape[1]-B.shape[1]))
                B = np.hstack((B,D))
                break
            if C == 0: 
                B = sameScale*A[:,:,k] + int(not sameScale)*normalize(A[:,:,k])
            else:
                B = np.hstack((B,sameScale*A[:,:,k] + int(not sameScale)*normalize(A[:,:,k])))
            k+=1
        if R == 0:
            CC = B 
        else:
            CC = np.vstack((CC,B))
        if k == A.shape[2]:
            break
    return normalize(CC)

def array3D_to_montage2D(A):
    return skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], grid_shape = (1,A.shape[2]), padding_width=0, fill=0)


from scipy.signal import medfilt2d # ------ for calculateVDP
plt.imshow(makeMontage(medfilt2d(Vent2.proton)))
plt.imshow(makeMontage(Vent2.proton))
plt.show()

RED = makeMontage(1*(Vent2.proton<(0.5*np.mean(Vent2.proton))))
BLUE = makeMontage(Vent2.proton)
GREEN = makeMontage(Vent2.mask)

plt.imshow(medfilt2d(RED,kernel_size = 5),cmap='gray')
plt.show()

plt.imshow(np.stack((RED,GREEN,BLUE),axis=2),cmap='gray')
plt.show()



import pywt
data = Vent2.mask[:,:,10]
coeffs = pywt.dwt2(data, 'haar')  # 'haar' is a common wavelet, you can try others

# The output coeffs is a tuple containing (approximation coefficients, (horizontal, vertical, diagonal coefficients))
cA, (cH, cV, cD) = coeffs

# Display the results
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].imshow(cA, cmap='gray')
axs[0, 0].set_title('Approximation Coefficients')
axs[0, 1].imshow(cH, cmap='gray')
axs[0, 1].set_title('Horizontal Detail Coefficients')
axs[1, 0].imshow(cV, cmap='gray')
axs[1, 0].set_title('Vertical Detail Coefficients')
axs[1, 1].imshow(cD, cmap='gray')
axs[1, 1].set_title('Diagonal Detail Coefficients')

plt.tight_layout()
plt.show()


threshold = 0.00000001

# Apply thresholding
def apply_threshold(arr, threshold):
    return np.where(np.abs(arr) > threshold, arr, 0)

# Apply threshold to the detail coefficients
cH_thresh = apply_threshold(cH, threshold)
cV_thresh = apply_threshold(cV, threshold)
cD_thresh = apply_threshold(cD, threshold)

# Reconstruct the data using the inverse 2D wavelet transform
filtered_coeffs = (cA, (cH_thresh, cV_thresh, cD_thresh))
reconstructed_data = pywt.idwt2(filtered_coeffs, 'haar')

# Plot the original, thresholded coefficients, and the reconstructed data
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Original data
axs[0].imshow(data, cmap='gray')
axs[0].set_title('Original Data')
# Reconstructed data after thresholding
axs[1].imshow(reconstructed_data, cmap='gray')
axs[1].set_title('Reconstructed Data')

# Display the difference (error) between original and reconstructed data
error = data - reconstructed_data
axs[2].imshow(error, cmap='gray')
axs[2].set_title('Difference (Error)')

plt.tight_layout()
plt.show()