from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# from NN_best_architecture import Net
# from NN import Net
# from NN_17_feb import Net
from NN_costum_loss import Net


from produce_plots_and_data import return_multiple_dipoles
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

N_samples = 10
N_dipoles = 2
name = 'multiple_dipoles'

# Amplitude
model = torch.load('trained_models/best_architecture_standarisation_w_amplitude_500_SGD_lr1.5_wd1e-05_bs32.pt')

# No amplitude
# model = torch.load('09_mai/09mai_adaptive_lr0.001_l1_11.april_NOTL2.pt')

# Two dipoles + amplitudes
# model = torch.load('trained_models/10000_11may_MSE_area_w_amplitude_150_SGD_lr1.5_wd0.1_mom0.35_bs64.pt')
# model = torch.load('trained_models/10000_11may_MSE_area_w_amplitude_1000_SGD_lr1.5_wd0.1_mom0.35_bs64.pt')
# print('finished loading model')

nyhead = NYHeadModel()

eeg, target = return_multiple_dipoles(N_samples, N_dipoles)
print('finished loading data')
# eeg, target = load_data_files(N_samples, name, num_dipoles=N_dipoles)

# target = np.reshape(target, (N_samples, 3*N_dipoles))
target = target.T.reshape(N_samples, 4*N_dipoles)

print(target)

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

        # # No amplitude
        # x_pred =  pred_list[dipole_num, i, 0] = pred[0]
        # y_pred =  pred_list[dipole_num, i, 1] = pred[1]
        # z_pred =  pred_list[dipole_num, i, 2] = pred[2]

        x_pred =  pred_list[dipole_num, i, 0] = denormalize(pred[0 + (dipole_num*4)], np.max(x_target), np.min(x_target))
        y_pred =  pred_list[dipole_num, i, 1] = denormalize(pred[1 + (dipole_num*4)], np.max(y_target), np.min(y_target))
        z_pred =  pred_list[dipole_num, i, 2] = denormalize(pred[2 + (dipole_num*4)], np.max(z_target), np.min(z_target))
        amplitude_pred = pred_list[dipole_num, i, 3] = denormalize(pred[3 + (dipole_num*4)], np.max(amplitude_target), np.min(amplitude_target))

        print(x_pred)
        print(y_pred)
        print(z_pred)
        print(amplitude_pred)
        input()

        relative_change_x[i] = relative_change(x_target[i], x_pred)
        relative_change_y[i] = relative_change(y_target[i], y_pred)
        relative_change_z[i] = relative_change(z_target[i], z_pred)
        relative_change_amplitude[i] = relative_change(amplitude_target[i], amplitude_pred)

        error_x[i] = np.abs(x_target[i] - x_pred)
        error_y[i] = np.abs(y_target[i] - y_pred)
        error_z[i] = np.abs(z_target[i] - z_pred)
        error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)

        amplitude_dict[amplitude_target[i]] = error_amplitude[i]

        # print(f'Sample {i}/{N_samples}')
        # print(f'True x-postion: {x_target[i]}')
        # print(f'Pred x-postion: {x_pred}')
        # print(f'Relative difference: {relative_change_x[i]}')
        # print(f'Absolute Error: {error_x[i]}')
        # print(' ')
        # print(f'True y-postion: {y_target[i]}')
        # print(f'Pred y-postion: {y_pred}')
        # print(f'Relative difference: {relative_change_y[i]}')
        # print(f'Absolute Error: {error_y[i]}')
        # print(' ')
        # print(f'True z-postion: {z_target[i]}')
        # print(f'Pred z-postion: {z_pred}')
        # print(f'Relative difference: {relative_change_z[i]}')
        # print(f'Absolute Error: {error_z[i]}')
        # print(' ')
        # print(f'True amplitude: {amplitude_target[i]}')
        # print(f'Pred amplitude: {amplitude_pred}')
        # print(f'Relative difference: {relative_change_amplitude[i]}')
        # print(f'Absolute Error: {error_amplitude[i]}')
        # print(' ')
        # print(' ')



for i in range(N_dipoles):
    print(f'Dipole No: {i+1}')
    print(f'MAE x-coordinates:{MAE(x_target, pred_list[i, :, 0])}')
    print(f'MAE y-coordinates:{MAE(y_target, pred_list[i, :, 1])}')
    print(f'MAE z-coordinates:{MAE(z_target, pred_list[i, :, 2])}')
    print(f'MAE amplitude:{MAE(amplitude_target, pred_list[i, :, 3])}')

# plot_mse_amplitude(amplitude_dict)
