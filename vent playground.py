import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('c:/vent_analysis/')
from Vent_Analysis import Vent_Analysis
import skimage.util # --------------------- for image montages

Vent2 = Vent_Analysis(pickle_path="C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_241006/Mepo0039_240301.pkl")
A = Vent2.HPvent

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

def get_CI_colorArrays(CI):
    CI[CI>40] = 40
    parula = np.load('C:\PIRL\data\parula.np.npy')
    CIred = parula[]

parula = np.load('C:\PIRL\data\parula.np.npy')

_, rr,cc,ss = Vent2.cropToData(Vent2.mask)

blank = np.zeros_like(Vent2.HPvent[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]])
proton = normalize(Vent2.proton[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]])
HP = normalize(Vent2.HPvent[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]])
N4 = normalize(Vent2.N4HPvent[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]])
border = normalize(Vent2.mask_border[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]])>0
defArr = Vent2.defectArray[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]]>0
CI = Vent2.CIarray[rr[0]:rr[-1],cc[0]:cc[-1],ss[0]:ss[-1]]

CIred = np.array([[[parula[int(CI[r,c,s]*64/40),0] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])
CIgreen = np.array([[[parula[int(CI[r,c,s]*64/40),1] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])
CIblue = np.array([[[parula[int(CI[r,c,s]*64/40),2] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])

RED3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 0*border, N4*(~defArr) + defArr, N4*(CI==0) + CIred*(CI>0)),axis=2)
GREEN3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 1*border, N4*(~defArr), N4*(CI==0) +CIgreen*(CI>0)),axis=2)
BLUE3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 1*border, N4*(~defArr), N4*(CI==0) +CIblue*(CI>0)),axis=2)

REDmontage = skimage.util.montage([RED3D[:,:,k] for k in range(0,RED3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
GREENmontage = skimage.util.montage([GREEN3D[:,:,k] for k in range(0,GREEN3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
BLUEmontage = skimage.util.montage([BLUE3D[:,:,k] for k in range(0,BLUE3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
IMAGE = np.stack((REDmontage,GREENmontage,BLUEmontage),axis=2)

plt.imshow(IMAGE)
for kk in ss:
    plt.text(-N4.shape[1]/2 + N4.shape[1]*kk,N4.shape[0]*2,f"{kk+1}",c='white')
plt.show()

