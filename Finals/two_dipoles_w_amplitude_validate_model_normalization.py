from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# from NN_dipole_w_amplitude import Net
from NN_2_dipoles import Net
# from NN_best_architecture import Net

from utils import numpy_to_torch, normalize, denormalize, MSE, MAE, relative_change, xz_plane_idxs
from load_data import load_data_files

import os
import h5py

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

def plot_mse_amplitude(amplitude_dict):
    labels = list(amplitude_dict.keys())
    values = list(amplitude_dict.values())

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'Relative difference', fontsize=20)
    ax.set_xlabel('Amplitude [mm]', fontsize=18)
    ax.set_ylabel('Error [mm]', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(labels, values)
    fig.savefig(f'plots/amplitude_mse_lr_1.8.png')

N_samples = 1000
N_dipoles = 2
name = 'dipole_w_amplitude'

# Amplitude
# model = torch.load('trained_models/dipole_w_amplitude__lr1.5_l1_1e-5_l2_1e-5_300_50000_bs32.pt')
# model = torch.load('trained_models/RIGHT_custom_area_w_amplitude_500_SGD_lr1.5_wd1e-05_bs32.pt')
# model = torch.load('trained_models/TEST_two_dipoles_w_amplitude_1000_SGD_lr0.1_wd0.25_mom0.35_bs128.pt')
# model = torch.load('trained_models/26june_two_dipoles_w_amplitude_5000_SGD_lr0.0001_wd0.1_mom0.35_bs128_10noise.pt')
model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.5_0_6000_(1).pt')


print('finished loading model')

nyhead = NYHeadModel()

eeg = np.load('data/dipoles_w_amplitudes_eeg_70000_1.npy')
target = np.load('data/dipoles_w_amplitudes_locations_70000_1.npy')

# eeg = np.load('data/multiple_dipoles_eeg_70000_2.npy')
# target = np.load('data/multiple_dipoles_locations_70000_2.npy')
print('finished loading data')

eeg = eeg[:N_samples*N_dipoles, :]
target = target[:, :N_samples*N_dipoles]

# target = target.reshape(N_samples, 4*N_dipoles)

target = target.T.reshape(N_samples, 4*N_dipoles)

eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg.T)

# pred_list = np.zeros((2, N_samples, 4))
pred_list = np.zeros((N_dipoles, N_samples, 4))

for dipole_num in range(N_dipoles):
    x_target = target[:, 0 + (dipole_num*4)]
    y_target = target[:, 1 + (dipole_num*4)]
    z_target = target[:, 2 + (dipole_num*4)]
    amplitude_target = target[:, 3 + (dipole_num*4)]

    error_x = np.zeros(N_samples)
    error_y = np.zeros_like(error_x)
    error_z = np.zeros_like(error_x)
    error_amplitude = np.zeros_like(error_x)

    relative_change_x = np.zeros(N_samples)
    relative_change_y = np.zeros_like(relative_change_x)
    relative_change_z = np.zeros_like(relative_change_x)
    relative_change_amplitude = np.zeros_like(relative_change_x)

    amplitude_dict = {key: None for key in amplitude_target}
    amplitude_dict = dict(sorted(amplitude_dict.items()))

    print(f'Dipole No: {dipole_num+1}')
    for i in range(N_samples):
        pred = model(eeg[:,i])
        pred = pred.detach().numpy()

        # denormalize target coordinates
        x_pred =  pred_list[dipole_num, i, 0] = denormalize(pred[0 + (dipole_num*4)], np.max(x_target), np.min(x_target))
        y_pred =  pred_list[dipole_num, i, 1] = denormalize(pred[1 + (dipole_num*4)], np.max(y_target), np.min(y_target))
        z_pred =  pred_list[dipole_num, i, 2] = denormalize(pred[2 + (dipole_num*4)], np.max(z_target), np.min(z_target))
        amplitude_pred = pred_list[dipole_num, i, 3] = denormalize(pred[3 + (dipole_num*4)], np.max(amplitude_target), np.min(amplitude_target))

        relative_change_x[i] = relative_change(x_target[i], x_pred)
        relative_change_y[i] = relative_change(y_target[i], y_pred)
        relative_change_z[i] = relative_change(z_target[i], z_pred)
        relative_change_amplitude[i] = relative_change(amplitude_target[i], amplitude_pred)

        error_x[i] = np.abs(x_target[i] - x_pred)
        error_y[i] = np.abs(y_target[i] - y_pred)
        error_z[i] = np.abs(z_target[i] - z_pred)
        error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)

        amplitude_dict[amplitude_target[i]] = error_amplitude[i]


for i in range(N_dipoles):
    print(f'Dipole No: {i+1}')
    print(f'MAE x-coordinates:{MAE(x_target, pred_list[i, :, 0])}')
    print(f'MAE y-coordinates:{MAE(y_target, pred_list[i, :, 1])}')
    print(f'MAE z-coordinates:{MAE(z_target, pred_list[i, :, 2])}')
    print(f'MAE amplitude:{MAE(amplitude_target, pred_list[i, :, 3])}')
# plot_mse_amplitude(amplitude_dict)
