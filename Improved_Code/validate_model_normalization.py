from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN import Net
from produce_plots_and_data import return_multiple_dipoles
from utils import numpy_to_torch, normalize, denormalize, MSE, relative_change, xz_plane_idxs

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
N_dipoles = 1

# model = torch.load('trained_models/25_april_(times8)_(6layers)_standarisation_w_amplitude_5000_SGD_lr0.001_wd1e-05.pt')
# model = torch.load('trained_models/best_architecture_standarisation_w_amplitude_500_SGD_lr1.5_wd1e-05_bs32.pt')
model = torch.load('trained_models/new_lr_standarisation_w_amplitude_500_SGD_lr1.8_wd1e-05_bs32.pt')


print('loading finished')

nyhead = NYHeadModel()

eeg, target = return_multiple_dipoles(N_samples, N_dipoles)

# eeg = normalize(eeg)
eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg.T)

x_target = target[0, :]
y_target = target[1, :]
z_target = target[2, :]
amplitude_target = target[-1, :]

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

pred_list = np.zeros((N_samples,4))

for i in range(N_samples):
    pred = model(eeg[:,i])
    pred = pred.detach().numpy()

    # denormalize target coordinates
    x_pred =  pred_list[i, 0] = denormalize(pred[0], np.max(x_target), np.min(x_target))
    y_pred =  pred_list[i, 1] = denormalize(pred[1], np.max(y_target), np.min(y_target))
    z_pred =  pred_list[i, 2] = denormalize(pred[2], np.max(z_target), np.min(z_target))

    # denormalize target amplitude
    amplitude_pred = pred_list[i, 3] = denormalize(pred[-1], np.max(amplitude_target), np.min(amplitude_target))

    relative_change_x[i] = relative_change(x_target[i], x_pred)
    relative_change_y[i] = relative_change(y_target[i], y_pred)
    relative_change_z[i] = relative_change(z_target[i], z_pred)
    relative_change_amplitude[i] = relative_change(amplitude_target[i], amplitude_pred)

    error_x[i] = np.abs(x_target[i] - x_pred)
    error_y[i] = np.abs(y_target[i] - y_pred)
    error_z[i] = np.abs(z_target[i] - z_pred)
    error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)

    amplitude_dict[amplitude_target[i]] = error_amplitude[i]

    print(f'Sample {i}/{N_samples}')
    print(f'True x-postion: {x_target[i]}')
    print(f'Pred x-postion: {x_pred}')
    print(f'Relative difference: {relative_change_x[i]}')
    print(f'Absolute Error: {error_x[i]}')
    print(' ')
    print(f'True y-postion: {y_target[i]}')
    print(f'Pred y-postion: {y_pred}')
    print(f'Relative difference: {relative_change_y[i]}')
    print(f'Absolute Error: {error_y[i]}')
    print(' ')
    print(f'True z-postion: {z_target[i]}')
    print(f'Pred z-postion: {z_pred}')
    print(f'Relative difference: {relative_change_z[i]}')
    print(f'Absolute Error: {error_z[i]}')
    print(' ')
    print(f'True amplitude: {amplitude_target[i]}')
    print(f'Pred amplitude: {amplitude_pred}')
    print(f'Relative difference: {relative_change_amplitude[i]}')
    print(f'Absolute Error: {error_amplitude[i]}')
    print(' ')
    print(' ')

print(f'MSE x-coordinates:{MSE(x_target, pred_list[:,0])}')
print(f'MSE y-coordinates:{MSE(y_target, pred_list[:,1])}')
print(f'MSE z-coordinates:{MSE(z_target, pred_list[:,2])}')
print(f'MSE amplitude:{MSE(amplitude_target, pred_list[:,3])}')

plot_mse_amplitude(amplitude_dict)
#
#
# plt.style.use('seaborn')
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
#
# def plot_mse_amplitude(amplitude_dict):
#     labels = list(amplitude_dict.keys())
#     values = list(amplitude_dict.values())
#
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.set_title(f'Relative difference', fontsize=20)
#     ax.set_xlabel('Amplitude [mm]', fontsize=18)
#     ax.set_ylabel('Error [mm]', fontsize=18)
#     ax.tick_params(axis='both', which='major', labelsize=18)
#     ax.plot(labels, values)
#     fig.savefig(f'plots/amplitude_mse_best.png')
#
# # N_samples = 1000
# # N_dipoles = 1
#
# # model = torch.load('trained_models/25_april_(times8)_(6layers)_standarisation_w_amplitude_5000_SGD_lr0.001_wd1e-05.pt')
# model = torch.load('trained_models/best_architecture_standarisation_w_amplitude_500_SGD_lr1.5_wd1e-05_bs32.pt')
# print('loading finished')
#
#
# nyhead = NYHeadModel()
#
# y_plane = 0
# # Plotting crossection of cortex
# threshold = 2  # threshold in mm for including points in plot
# #inneholder alle idekser i koreks i planet mitt
# xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - y_plane) < threshold)[0]
#
# # eeg, target = return_multiple_dipoles(N_samples, N_dipoles)
#
# eeg, x_pred_list, y_pred_list, z_pred_list = xz_plane_idxs()
#
# # eeg = normalize(eeg)
# eeg = (eeg - np.mean(eeg))/np.std(eeg)
# eeg = numpy_to_torch(eeg.T)
#
# x_target = target[0, :]
# y_target = target[1, :]
# z_target = target[2, :]
# amplitude_target = target[-1, :]
#
# error_x = np.zeros(N_samples)
# error_y = np.zeros_like(error_x)
# error_z = np.zeros_like(error_x)
# error_amplitude = np.zeros_like(error_x)
#
# relative_change_x = np.zeros(N_samples)
# relative_change_y = np.zeros_like(relative_change_x)
# relative_change_z = np.zeros_like(relative_change_x)
# relative_change_amplitude = np.zeros_like(relative_change_x)
#
#
# amplitude_dict = {key: None for key in amplitude_target}
# amplitude_dict = dict(sorted(amplitude_dict.items()))
#
# pred_list = np.zeros((N_samples,4))
#
# for i in range(len(xz_plane_idxs)):
#     pred = model(eeg[:,i])
#     pred = pred.detach().numpy()
#
#     # denormalize target coordinates
#     x_pred =  pred_list[i, 0] = denormalize(pred[0], np.max(x_target), np.min(x_target))
#     y_pred =  pred_list[i, 1] = denormalize(pred[1], np.max(y_target), np.min(y_target))
#     z_pred =  pred_list[i, 2] = denormalize(pred[2], np.max(z_target), np.min(z_target))
#
#     # denormalize target amplitude
#     amplitude_pred = pred_list[i, 3] = denormalize(pred[-1], np.max(amplitude_target), np.min(amplitude_target))
#
#     relative_change_x[i] = relative_change(x_target[i], x_pred)
#     relative_change_y[i] = relative_change(y_target[i], y_pred)
#     relative_change_z[i] = relative_change(z_target[i], z_pred)
#     relative_change_amplitude[i] = relative_change(amplitude_target[i], amplitude_pred)
#
#     error_x[i] = np.abs(x_target[i] - x_pred)
#     error_y[i] = np.abs(y_target[i] - y_pred)
#     error_z[i] = np.abs(z_target[i] - z_pred)
#     error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)
#
#     amplitude_dict[amplitude_target[i]] = error_amplitude[i]
#
#     print(f'Sample {i}/{N_samples}')
#     print(f'True x-postion: {x_target[i]}')
#     print(f'Pred x-postion: {x_pred}')
#     print(f'Relative difference: {relative_change_x[i]}')
#     print(f'Absolute Error: {error_x[i]}')
#     print(' ')
#     print(f'True y-postion: {y_target[i]}')
#     print(f'Pred y-postion: {y_pred}')
#     print(f'Relative difference: {relative_change_y[i]}')
#     print(f'Absolute Error: {error_y[i]}')
#     print(' ')
#     print(f'True z-postion: {z_target[i]}')
#     print(f'Pred z-postion: {z_pred}')
#     print(f'Relative difference: {relative_change_z[i]}')
#     print(f'Absolute Error: {error_z[i]}')
#     print(' ')
#     print(f'True amplitude: {amplitude_target[i]}')
#     print(f'Pred amplitude: {amplitude_pred}')
#     print(f'Relative difference: {relative_change_amplitude[i]}')
#     print(f'Absolute Error: {error_amplitude[i]}')
#     print(' ')
#     print(' ')
#
# print(f'MSE x-coordinates:{MSE(x_target, pred_list[:,0])}')
# print(f'MSE y-coordinates:{MSE(y_target, pred_list[:,1])}')
# print(f'MSE z-coordinates:{MSE(z_target, pred_list[:,2])}')
# print(f'MSE amplitude:{MSE(amplitude_target, pred_list[:,3])}')
#
# plot_mse_amplitude(amplitude_dict)







