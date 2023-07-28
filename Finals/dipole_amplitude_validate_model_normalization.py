from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# from NN_dipole_w_amplitude import Net
# from NN_costum_loss import Net
# from NN_simple_network_amplitude import Net
from ffnn import FFNN


from utils import numpy_to_torch, normalize, denormalize, MSE, MAE, xz_plane_idxs
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
    # fig.savefig(f'plots/.png')

N_dipoles = 2
name = 'dipole_w_amplitude'

model = torch.load('trained_models/amplitudes_64_0.001_0.35_0.1_1e-05_3000_(1).pt')

print('finished loading model')

nyhead = NYHeadModel()

eeg = np.load('data/amplitudes_70000_2_eeg_test.npy')
target = np.load('data/amplitudes_70000_2_targets_test.npy')
print('finished loading data')

# target = target.reshape(N_samples, 4*N_dipoles)
N_samples = 20000
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

        error_x[i] = np.abs(x_target[i] - x_pred)
        error_y[i] = np.abs(y_target[i] - y_pred)
        error_z[i] = np.abs(z_target[i] - z_pred)
        error_amplitude[i] = np.abs(amplitude_target[i] - amplitude_pred)

        amplitude_dict[amplitude_target[i]] = error_amplitude[i]


for i in range(N_dipoles):
    print(x_target.shape)
    print(pred_list[i, :, 0].shape)
    MAE_x = MAE(x_target, pred_list[i, :, 0])
    MAE_y = MAE(y_target, pred_list[i, :, 1])
    MAE_z = MAE(z_target, pred_list[i, :, 2])
    MAE_amp = MAE(amplitude_target, pred_list[i, :, 3])
    MAE_pos = MAE(target, pred_list)
    #
    MSE_x = MSE(x_target, pred_list[i, :, 0])
    MSE_y = MSE(y_target, pred_list[i, :, 1])
    MSE_z = MSE(z_target, pred_list[i, :, 2])
    MAE_amp = MAE(amplitude_target, pred_list[i, :, 3])
    MSE_pos = MSE(target, pred_list)

    # relative_change_pos = np.mean(relative_change(target, pred_list))

    print(f'Dipole No: {i+1}')
    print(f'MAE x-coordinates:{MAE_x} mm')
    print(f'MAE y-coordinates:{MAE_y} mm')
    print(f'MAE z-coordinates:{MAE_z} mm')
    print(f'MAE amplitude:{MAE_amp} mm')
    print(f'MAE: {MAE_pos} mm')

    print(f'Dipole No: {i+1}')
    print(f'MSE x-coordinates:{MSE_x} mm')
    print(f'MSE y-coordinates:{MSE_y} mm')
    print(f'MSE y-coordinates:{MSE_z} mm')
    print(f'MSE amplitude:{MAE_amp} mm')
    print(f'MSE: {MSE_pos} mm')

    print(f'Dipole No: {i+1}')
    print(f'RMSE x-coordinates:{np.sqrt(MSE(x_target, pred_list[i, :, 0]))}')
    print(f'RMSE y-coordinates:{np.sqrt(MSE(y_target, pred_list[i, :, 1]))}')
    print(f'RMSE z-coordinates:{np.sqrt(MSE(z_target, pred_list[i, :, 2]))}')
    print(f'RMSE amplitude:{np.sqrt(MSE(amplitude_target, pred_list[i, :, 3]))}')
    print(f'RMSE: {np.sqrt(MSE_pos)} mm')

