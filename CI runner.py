import Vent_Analysis as vent
import pyreadr # -- also need 'xarray' installed to use

A = pyreadr.read_r('//umh.edu/data/Radiology/Xenon_Studies/Studies/General_Xenon/Gen_Xenon_Studies/Xe-0055 - 230126/PreSildenafil/VDPBL.RData')


Vent1 = vent.Vent_Analysis(xenon_array = A['HP'].to_numpy(),mask_array = A['Mask'].to_numpy())
Vent1.runVDP()
Vent1.VDP
Vent1.vox = [3,3,3]
Vent1.calculate_CI()
Vent1.CI