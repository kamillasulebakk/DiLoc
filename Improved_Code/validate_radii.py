from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# from new_dipole_w_radi_amplitude import Net
from NN_costum_loss import Net

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
    fig.savefig(f'plots/dipole_w_radi_amplitude_mse_lr_1.5.png')

# eeg, target = load_data_files(50000, 'dipole_area', num_dipoles=1)

eeg = np.load('data/validate_const_A_dipole_area_const_A_eeg_70000_1.npy')
target = np.load('data/validate_const_A_dipole_area_const_A_locations_70000_1.npy')


print('finished loading data')

N_samples = 1000

eeg = eeg[:N_samples,:]
target = target[:N_samples,:]

# model = torch.load('trained_models/50000_23june_MSE_area_w_amplitude_500_SGD_lr1.5_wd0.1_mom0.35_bs64.pt')
# model = torch.load('trained_models/50000_23junemseloss_MSE_area_w_amplitude_500_SGD_lr1.5_wd0.1_mom0.35_bs64.pt')
# model = torch.load('trained_models/50000_26junemseloss_MSE_area_w_amplitude_500_SGD_lr_0.001_wd0.1_mom0.35_bs64.pt')
model = torch.load('trained_models/50000_26junemseloss_MSE_area_w_amplitude_5000_SGD_lr0.001_wd0.1_mom0.35_bs64.pt')


print('loading finished')

nyhead = NYHeadModel()
N_samples = 1000

eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg.T)

x_target = target[:, 0]
y_target = target[:, 1]
z_target = target[:, 2]
radius_target = target[:, -2]
amplitude_target = target[:, -1]

error_x = np.zeros(N_samples)
error_y = np.zeros_like(error_x)
error_z = np.zeros_like(error_x)
error_radius = np.zeros_like(error_x)
error_amplitude = np.zeros_like(error_x)

relative_change_x = np.zeros(N_samples)
relative_change_y = np.zeros_like(relative_change_x)
relative_change_z = np.zeros_like(relative_change_x)
relative_change_radius = np.zeros_like(relative_change_x)
relative_change_amplitude = np.zeros_like(relative_change_x)


amplitude_dict = {key: None for key in amplitude_target}
amplitude_dict = dict(sorted(amplitude_dict.items()))

pred_list = np.zeros((N_samples,5))

for i in range(N_samples):
    pred = model(eeg[:,i])
    pred = pred.detach().numpy()

    # denormalize target coordinates
    x_pred =  pred_list[i, 0] = denormalize(pred[0], np.max(x_target), np.min(x_target))
    y_pred =  pred_list[i, 1] = denormalize(pred[1], np.max(y_target), np.min(y_target))
    z_pred =  pred_list[i, 2] = denormalize(pred[2], np.max(z_target), np.min(z_target))
    radius_pred = pred_list[i, 3] = denormalize(pred[-2], np.max(radius_target), np.min(radius_target))
    amplitude_pred = pred_list[i, 4] = denormalize(pred[-1], np.max(amplitude_target), np.min(amplitude_target))

    relative_change_x[i] = relative_change(x_target[i], x_pred)
    relative_change_y[i] = relative_change(y_target[i], y_pred)
    relative_change_z[i] = relative_change(z_target[i], z_pred)
    relative_change_radius[i] = relative_change(radius_target[i], radius_pred)
    relative_change_amplitude[i] = relative_change(amplitude_target[i], amplitude_pred)

    error_x[i] = np.abs(x_target[i] - x_pred)
    error_y[i] = np.abs(y_target[i] - y_pred)
    error_z[i] = np.abs(z_target[i] - z_pred)
    error_radius[i] = np.abs(radius_target[i] - radius_pred)
    error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)

    amplitude_dict[amplitude_target[i]] = error_amplitude[i]

print(f'Relative error x-coordinates:{MAE(x_target, pred_list[:,0])}')
print(f'Relative error y-coordinates:{MAE(y_target, pred_list[:,1])}')
print(f'Relative error z-coordinates:{MAE(z_target, pred_list[:,2])}')
print(f'Relative error radius:{MAE(radius_target, pred_list[:,-2])}')
print(f'Relative error amplitude:{MAE(amplitude_target, pred_list[:,-1])}')

plot_mse_amplitude(amplitude_dict)









