from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_pos_and_radii import Net
from produce_and_load_eeg_data import return_dipole_area, load_mean_std
from utils import numpy_to_torch

import os
import h5py

model = torch.load('trained_models/NN_pos_and_radii_noise10_10000.pt')

# 15 what? mm?
eeg, dipole_locations_and_radii = return_dipole_area(10, 15)

eeg = numpy_to_torch(eeg)

pred = model(eeg).T

print(pred)
print(dipole_locations_and_radii)
input()

mse = np.zeros((100, 4))

for i in range(100):
    # for j in range(4):
    print(dipole_locations_and_radii[:-1][i])
    print(pred[:-1,i])

    print(dipole_locations_and_radii[3][i])
    print(pred[3][i])
    mse[i][3] = np.mean((dipole_locations_and_radii[3][i] - pred[3][i].detach().numpy())**2)


print(mse)

#
#
# from lfpykit.eegmegcalc import NYHeadModel
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# from NN_pos_and_radii import Net
# from produce_and_load_eeg_data import return_dipole_area, load_mean_std
# from utils import numpy_to_torch
#
# import os
# import h5py
#
# model = torch.load('trained_models/NN_pos_and_radii_noise10_10000.pt')
# eeg, dipole_locations_and_radii = return_dipole_area(10, 15)
# eeg = numpy_to_torch(eeg)
#
# pred = model(eeg).T
# pred = pred.detach().numpy()
#
# print(dipole_locations_and_radii)
# print(pred)
# input()
#
# #De-normalize
# pred_locations = (pred[:-1,:] + np.mean(pred[:-1,:-1]))*np.std(pred[:-1,:])
# pred_radii = (pred[-3,:] + np.mean(pred[-3,:]))*np.std(pred[-3,:])
#
# print(pred_locations)
# print(pred_radii)
#
# input()
#
# mse_locations = np.zeros((10, 3))
# mse_radii = np.zeros(10)
#
# for i in range(10):
#     for j in range(3):
#         mse_locations[i][j] = np.mean((dipole_locations_and_radii[j][i] - pred_locations[j][i])**2)
#
# for i in range(10):
#     mse_radii[i] = np.mean((dipole_locations_and_radii[3][i] - pred_radii[i])**2)
#
# print(mse_radii)
# print(mse_locations)

