from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN
from eeg_dataset import EEGDataset

from produce_data import calculate_eeg, return_dipole_population_indices
from utils import numpy_to_torch, normalize, denormalize, MSE, MAE
import matplotlib as mpl

import os
import h5py

plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


import matplotlib.gridspec as gridspec

def plot_MSE_error(MSE, dipole_locs, name, numbr):
    fig = plt.figure(figsize=[10, 8])  # Increase the figure size

    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    cax = fig.add_axes([0.92, 0.55, 0.01, 0.3])  # This axis is just the colorbar

    scatter_params = dict(cmap="hot", vmin=0, vmax=15, s=10)

    if numbr == 0:
        fig.suptitle(f'MSE for dipole locations in y cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[2], c=MSE, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=16)
        ax.set_ylabel("z [mm]", fontsize=16)
    elif numbr == 1:
        fig.suptitle(f'MSE for dipole locations in z cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[1], c=MSE, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=16)
        ax.set_ylabel("y [mm]", fontsize=16)
    else:
        fig.suptitle(f'MSE for dipole locations in x cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[1], dipole_locs[2], c=MSE, **scatter_params)
        ax.set_xlabel("y [mm]", fontsize=16)
        ax.set_ylabel("z [mm]", fontsize=16)

    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel('[mm]', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.label.set_fontsize(16)  # Set x-label font size
    ax.yaxis.label.set_fontsize(16)  # Set y-label font size

    plt.savefig(f"plots/MSE_dipole_area_{name}_{numbr}.pdf")

data = EEGDataset()
model = torch.load('trained_models/area_32_0.001_0.35_0.1_0.0_5000_(0).pt')

eeg = np.load('data/area_70000_1_eeg_test.npy')
eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg)
target = np.load('data/area_70000_1_targets_test.npy')
print('Test data loaded')

nyhead = NYHeadModel()
sulci_map = np.array(nyhead.head_data["cortex75K"]["sulcimap"], dtype=int)[0]


x_plane = 10
y_plane = 0
z_plane = 0

# Plotting crossection of cortex
threshold = 2  # threshold in mm for including points in plot

#inneholder alle idekser i koreks i planet mitt
xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - y_plane) < threshold)[0]
xy_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] - z_plane) < threshold)[0]
yz_plane_idxs = np.where(np.abs(nyhead.cortex[0, :] - x_plane) < threshold)[0]

planes = [xz_plane_idxs, xy_plane_idxs, yz_plane_idxs]

sulci_error_location = []
gyri_error_location = []

for numbr, plane in enumerate(planes):
    error_locations = []
    error_x = []
    error_y = []
    error_z = []

    pred_list = np.zeros((len(plane),3))
    x_target_list = np.zeros(len(plane))
    y_target_list = np.zeros(len(plane))
    z_target_list = np.zeros(len(plane))


    for i, idx in enumerate(plane):
        num_samples = len(nyhead.cortex[:,idx])

        eeg = np.zeros((num_samples, 231))

        A = 10/899 # max total amplitude for dipole population is 10

        for i in range(num_samples):
            pos_indices = return_dipole_population_indices(
                nyhead, dipole_centers[:,i], 10
            )

            # ensure that population consists of at least one dipole
            while len(pos_indices) < 1:
                radii[i] += 1
                pos_idx = return_dipole_population_indices(nyhead, dipole_centers[:,i], radii[i])

            dipole_amplitudes[i] = A*len(pos_indices)
            for idx in pos_indices:
                nyhead.set_dipole_pos(nyhead.cortex[:, idx])
                eeg[i] += calculate_eeg(nyhead, A)

        eeg = (eeg - np.mean(eeg))/np.std(eeg)
        eeg = numpy_to_torch(eeg.T)

        pred = model(eeg)
        pred = pred.detach().numpy().flatten()

        x_target = x_target_list[i] = nyhead.cortex[0,idx]
        y_target = y_target_list[i] = nyhead.cortex[1,idx]
        z_target = z_target_list[i] = nyhead.cortex[2,idx]

        pred_list[i, 0] = denormalize(pred[0], data.max_targets[0], data.min_targets[0])
        pred_list[i, 1] = denormalize(pred[1], data.max_targets[1], data.min_targets[2])
        pred_list[i, 2] = denormalize(pred[2], data.max_targets[2], data.min_targets[2])

        error_i_x = MSE(x_target, x_pred)
        error_x.append(error_i_x)

        error_i_y = MSE(y_target, y_pred)
        error_y.append(error_i_y)

        error_i_z = MSE(z_target, z_pred)
        error_z.append(error_i_z)

        error_i_locations = (error_i_x + error_i_y + error_i_z)/3
        error_locations.append(error_i_locations)

        if sulci_map[idx] == 1:
            sulci_error_location.append(error_i_locations)
        else:
            gyri_error_location.append(error_i_locations)


    target = np.concatenate((x_target_list, y_target_list, z_target_list))
    target = target.reshape((np.shape(pred_list)[1], np.shape(pred_list)[0]))
    target = target.T

    print(f'MSE x-coordinates:{error_x}')
    print(f'MSE y-coordinates:{error_y}')
    print(f'MSE z-coordinates:{error_z}')
    print(f'MSE location:{error_locations}')
    print('')

    plot_MSE_error(error_locations, nyhead.cortex[:,plane], 'Euclidean Distance', numbr)


print(np.mean(sulci_error_location))
print(np.mean(gyri_error_location))

# plot_MSE_error(error_x, nyhead.cortex[:,plane], 'x', numbr)
# plot_MSE_error(error_y, nyhead.cortex[:,plane], 'y', numbr)
# plot_MSE_error(error_z, nyhead.cortex[:,plane], 'z', numbr)




