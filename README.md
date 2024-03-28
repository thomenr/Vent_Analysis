# PIRL Ventilation Analysis Pipeline, RPT, 3/27/2024
This is the Ventilation analysis pipeline for MU data, version 240327_RPT.

## Overview
There's 2 important pieces to this code. First, this contains the *Vent_Analysis* class which contains the processing pipeline for analyzing xenon ventilation data given the xenon image set (a DICOM file) and corresponding segmentation (a folder containing mask DICOMs). Second, the __main__ script employs the PySimpleGUI module to create a graphical user interface [GUI] so the data can be processed easily. Here's what the GUI looks like:
![alt text](https://github.com/thomenr/Vent_Analysis/blob/main/GUI.png)
The [powerpoint](https://github.com/thomenr/Vent_Analysis/blob/main/Vent_Analysis.pptx) gives an overview of how it all works. Essentially, you just plug in paths to your data and click the buttons for methods you want to run. Once they've run, the image windows will update with your data and you can export the analyses. Easy peasy!

## Setup
Using your favorite [git bash](https://git-scm.com/downloads) clone this github:  
`git clone https://github.com/thomenr/Vent_Analysis`  
Next you'll need to install the python modules listed in the [requirements.txt](https://github.com/thomenr/Vent_Analysis/blob/main/requirements.txt) file:  
`pip install -r requirements.txt`  
Now, using your favorite [IDE](https://code.visualstudio.com/download), run the code (see the powerpoint for a few ways to do this - slide 3).
Follow the GUI instructions and you'll be calculating VDPs in no time!

## Things to do in future versions
 - Vent_Analysis class inputs either paths to data or the arrays themselves
 - Output DICOM header info as JSON
 - get more header info (both TWIX and DICOM) into metadata variable
 - CI colormap output in screenshot
 - Multiple VDPs calculated (linear binning, k-means)
 - show histogram?
 - edit mask
 - automatic segmentation using proton (maybe DL this?)
 - Denoise Option