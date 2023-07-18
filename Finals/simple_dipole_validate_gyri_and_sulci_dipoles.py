from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_simple_dipole import Net

from produce_plots_and_data import calculate_eeg
from utils import numpy_to_torch, normalize, denormalize, MSE, MAE
import produce_plots_and_data
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

def plot_MAE_error(mse, dipole_locs, name, numbr):
    fig = plt.figure(figsize=[10, 8])  # Increase the figure size

    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    cax = fig.add_axes([0.92, 0.55, 0.01, 0.3])  # This axis is just the colorbar

    scatter_params = dict(cmap="hot", vmin=0, vmax=15, s=10)

    if numbr == 0:
        fig.suptitle(f'MAE for dipole locations in y cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[2], c=mse, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=16)
        ax.set_ylabel("z [mm]", fontsize=16)
    elif numbr == 1:
        fig.suptitle(f'MAE for dipole locations in z cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[1], c=mse, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=16)
        ax.set_ylabel("y [mm]", fontsize=16)
    else:
        fig.suptitle(f'MAE for dipole locations in x cross-section', fontsize=20)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[1], dipole_locs[2], c=mse, **scatter_params)
        ax.set_xlabel("y [mm]", fontsize=16)
        ax.set_ylabel("z [mm]", fontsize=16)

    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.set_ylabel('[mm]', fontsize=16)
    cbar.ax.tick_params(labelsize=16)

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.label.set_fontsize(16)  # Set x-label font size
    ax.yaxis.label.set_fontsize(16)  # Set y-label font size

    plt.savefig(f"plots/NEW_simple_dipole_error_{name}_{numbr}.pdf")

# x, y, z - coordinates
model = torch.load('trained_models/simple_dipole_lr0.001_RELU_500_50000.pt')

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

sulci_error = []
gyri_error = []

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
        nyhead.set_dipole_pos(nyhead.cortex[:,idx])
        eeg = calculate_eeg(nyhead)
        eeg = (eeg - np.mean(eeg))/np.std(eeg)
        eeg = numpy_to_torch(eeg.T)
        pred = model(eeg)
        pred = pred.detach().numpy().flatten()


        x_pred = pred_list[i, 0] = pred[0]
        y_pred = pred_list[i, 1] = pred[1]
        z_pred = pred_list[i, 2] = pred[2]

        x_target = x_target_list[i] = nyhead.cortex[0,idx]
        y_target = y_target_list[i] = nyhead.cortex[1,idx]
        z_target = z_target_list[i] = nyhead.cortex[2,idx]

        error_i_x = np.abs(x_target - x_pred)
        error_x.append(error_i_x)

        error_i_y = np.abs(y_target - y_pred)
        error_y.append(error_i_y)

        error_i_z = np.abs(z_target - z_pred)
        error_z.append(error_i_z)

        error_i_locations = (error_i_x + error_i_y + error_i_z)/3
        error_locations.append(error_i_locations)

        if sulci_map[idx] == 1:
            sulci_error.append(error_i_locations)
        else:
            gyri_error.append(error_i_locations)


    target = np.concatenate((x_target_list, y_target_list, z_target_list))
    target = target.reshape((np.shape(pred_list)[1], np.shape(pred_list)[0]))
    target = target.T

    MAE_x = MAE(x_target_list, pred_list[:,0])
    MAE_y = MAE(y_target_list, pred_list[:,1])
    MAE_z = MAE(z_target_list, pred_list[:,2])
    MAE_locations = MAE(target, pred_list)

    print(f'MAE x-coordinates:{MAE_x}')
    print(f'MAE y-coordinates:{MAE_y}')
    print(f'MAE z-coordinates:{MAE_z}')
    print(f'MAE location:{MAE_locations}')


print(np.mean(sulci_error))
print(np.mean(gyri_error))
    # plot_MAE_error(error_x, nyhead.cortex[:,plane], 'x', numbr)
    # plot_MAE_error(error_y, nyhead.cortex[:,plane], 'y', numbr)
    # plot_MAE_error(error_z, nyhead.cortex[:,plane], 'z', numbr)
    # plot_MAE_error(error_locations, nyhead.cortex[:,plane], 'Euclidean Distance', numbr)




