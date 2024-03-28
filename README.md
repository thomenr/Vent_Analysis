# PIRL Ventilation Analysis Pipeline, RPT, 3/27/2024
This is the Ventilation analysis pipeline for MU data, version 240327_RPT.

## Overview
There's 2 important pieces to this code. First, this contains the *Vent_Analysis* class which contains the processing pipeline for analyzing xenon ventilation data given the xenon image set (a DICOM file) and corresponding segmentation (a folder containing mask DICOMs). Second, the __main__ script employs the PySimpleGUI module to create a graphical user interface [GUI] so the data can be processed easily. Here's what the GUI looks like:
![alt text](https://github.com/thomenr/Vent_Analysis/blob/main/GUI.png)
The [powerpoint](https://github.com/thomenr/Vent_Analysis/blob/main/Vent_Analysis.pptx) gives an overview of how it all works. Essentially, you just plug in paths to your data and click the buttons for methods you want to run. Once they've run, the image windows will update with your data and you can export the analyses. Easy peasy!

## Setup
Clone this github:
`<code>` git clone https://github.com/thomenr/Vent_Analysis
Hooray