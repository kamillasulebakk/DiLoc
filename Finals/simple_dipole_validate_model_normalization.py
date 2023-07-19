from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_simple_dipole import Net
# from NN_less_complicated_network_radius import Net

from utils import numpy_to_torch, normalize, denormalize, MSE, MAE, xz_plane_idxs
from load_data import load_data_files

import os
import h5py

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

N_samples = 1000
N_dipoles = 1
name = 'simple_dipole'

# model = torch.load('trained_models/TEST_simple_dipole_lr0.001_l1_penalty_300_50000.pt')
model = torch.load('trained_models/simple_dipole_lr0.001_RELU_500_50000.pt')
# model = torch.load('trained_models/july/simple_dipole_l2_less_complicated_network_radius_tanh_50000_19july_mseloss_MSE_dipole_w_amplitude_500_SGD_lr0.001_mom0.35_wd_0_bs64.pt')


print('finished loading model')

nyhead = NYHeadModel()

eeg = np.load('data/validate_simple_dipole_eeg_70000_1.npy')
target = np.load('data/validate_simple_dipole_locations_70000_1.npy')
print('finished loading data')

eeg = eeg[:N_samples,:]
target = target[:N_samples,:]

target = np.reshape(target, (N_samples, 3*N_dipoles))

eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg.T)

# pred_list = np.zeros((2, N_samples, 4))
pred_list = np.zeros((N_dipoles, N_samples, 3))

for dipole_num in range(N_dipoles):
    x_target = target[:, 0 + (dipole_num*3)]
    y_target = target[:, 1 + (dipole_num*3)]
    z_target = target[:, 2 + (dipole_num*3)]

    error_x = np.zeros(N_samples)
    error_y = np.zeros_like(error_x)
    error_z = np.zeros_like(error_x)

    # relative_change_x = np.zeros(N_samples)
    # relative_change_y = np.zeros_like(relative_change_x)
    # relative_change_z = np.zeros_like(relative_change_x)

    print(f'Dipole No: {dipole_num+1}')
    for i in range(N_samples):
        pred = model(eeg[:,i])
        pred = pred.detach().numpy()

        x_pred =  pred_list[dipole_num, i, 0] = pred[0]
        y_pred =  pred_list[dipole_num, i, 1] = pred[1]
        z_pred =  pred_list[dipole_num, i, 2] = pred[2]

        # relative_change_x[i] = relative_change(x_target[i], x_pred)
        # relative_change_y[i] = relative_change(y_target[i], y_pred)
        # relative_change_z[i] = relative_change(z_target[i], z_pred)

        error_x[i] = np.abs(x_target[i] - x_pred)
        error_y[i] = np.abs(y_target[i] - y_pred)
        error_z[i] = np.abs(z_target[i] - z_pred)



for i in range(N_dipoles):
    MAE_x = MAE(x_target, pred_list[i, :, 0])
    MAE_y = MAE(y_target, pred_list[i, :, 1])
    MAE_z = MAE(z_target, pred_list[i, :, 2])
    MAE_pos = MAE(target, pred_list)
    #
    MSE_x = MSE(x_target, pred_list[i, :, 0])
    MSE_y = MSE(y_target, pred_list[i, :, 1])
    MSE_z = MSE(z_target, pred_list[i, :, 2])
    MSE_pos = MSE(target, pred_list)

    # relative_change_pos = np.mean(relative_change(target, pred_list))

    print(f'Dipole No: {i+1}')
    print(f'MAE x-coordinates:{MAE_x} mm')
    print(f'MAE y-coordinates:{MAE_y} mm')
    print(f'MAE z-coordinates:{MAE_z} mm')
    print(f'MAE: {MAE_pos} mm')

    print(f'Dipole No: {i+1}')
    print(f'MAE x-coordinates:{MSE_x} mm')
    print(f'MAE y-coordinates:{MSE_y} mm')
    print(f'MAE z-coordinates:{MSE_z} mm')
    print(f'MAE: {MSE_pos} mm')

    print(f'Dipole No: {i+1}')
    print(f'RMSE x-coordinates:{np.sqrt(MSE(x_target, pred_list[i, :, 0]))}')
    print(f'RMSE y-coordinates:{np.sqrt(MSE(y_target, pred_list[i, :, 1]))}')
    print(f'RMSE z-coordinates:{np.sqrt(MSE(z_target, pred_list[i, :, 2]))}')
    print(f'RMSE z-coordinates: {np.sqrt(MSE_pos)} mm')
