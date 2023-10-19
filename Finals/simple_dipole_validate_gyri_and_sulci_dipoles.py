from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN

from produce_data import calculate_eeg
from utils import numpy_to_torch, normalize, denormalize, MSE, MSE
import produce_data
import matplotlib as mpl

from plot import set_ax_info

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


def plot_MED_error(med, dipole_locs, name, numbr):
    fig = plt.figure(figsize=[10, 8])  # Increase the figure size

    fig.subplots_adjust(hspace=0.4, left=0.07, right=0.9, bottom=0.1, top=0.85)

    # cax = fig.add_axes([0.92, 0.55, 0.01, 0.3])  # This axis is just the colorbar

    scatter_params = dict(cmap="hot", vmin=0, vmax=15, s=12)

    if numbr == 0:
        fig.suptitle(f'ED for Dipole Locations in the X-Z Cross-Section', fontsize=30)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[2], c=med, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=25)
        ax.set_ylabel("z [mm]", fontsize=25)
    elif numbr == 1:
        fig.suptitle(f'ED for Dipole Locations in the X-Y Cross-Section', fontsize=30)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[0], dipole_locs[1], c=med, **scatter_params)
        ax.set_xlabel("x [mm]", fontsize=25)
        ax.set_ylabel("y [mm]", fontsize=25)
    else:
        fig.suptitle(f'ED for Dipole Locations in the Y-Z Cross-Section', fontsize=30)
        ax = fig.add_subplot(111, aspect=1)
        img = ax.scatter(dipole_locs[1], dipole_locs[2], c=med, **scatter_params)
        ax.set_xlabel("y [mm]", fontsize=25)
        ax.set_ylabel("z [mm]", fontsize=25)

    # cbar = plt.colorbar(img, cax=cax)
    # cbar.ax.set_ylabel('[mm]', fontsize=25)
    # cbar.ax.tick_params(labelsize=25)

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.xaxis.label.set_fontsize(25)  # Set x-label font size
    ax.yaxis.label.set_fontsize(25)  # Set y-label font size

    plt.savefig(f"plots/Simple/MED_simple_dipole_error_{name}_{numbr}.pdf")

# x, y, z - coordinates
model = torch.load('trained_models/simple_32_0.001_0.35_0.5_0.0_500_(0).pt')

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
        eeg = np.expand_dims(eeg, axis=1)
        eeg = numpy_to_torch(eeg)
        eeg = eeg.flatten()
        pred = model(eeg)
        pred = pred.detach().numpy().flatten()


        x_pred = pred_list[i, 0] = pred[0]
        y_pred = pred_list[i, 1] = pred[1]
        z_pred = pred_list[i, 2] = pred[2]

        x_target = x_target_list[i] = nyhead.cortex[0,idx]
        y_target = y_target_list[i] = nyhead.cortex[1,idx]
        z_target = z_target_list[i] = nyhead.cortex[2,idx]

        error_i_x = (x_target - x_pred) ** 2
        error_x.append(error_i_x)

        error_i_y = (y_target - y_pred) ** 2
        error_y.append(error_i_y)

        error_i_z = (z_target - z_pred) ** 2
        error_z.append(error_i_z)

        error_i_locations = np.sqrt((error_i_x + error_i_y + error_i_z))
        error_locations.append(error_i_locations)

        if sulci_map[idx] == 1:
            sulci_error.append(error_i_locations)
        else:
            gyri_error.append(error_i_locations)


    target = np.concatenate((x_target_list, y_target_list, z_target_list))
    target = target.reshape((np.shape(pred_list)[1], np.shape(pred_list)[0]))
    target = target.T

    MSE_x = MSE(x_target_list, pred_list[:,0])
    MSE_y = MSE(y_target_list, pred_list[:,1])
    MSE_z = MSE(z_target_list, pred_list[:,2])
    MSE_locations = MSE(target, pred_list)

    print(f'MSE x-coordinates:{MSE_x}')
    print(f'MSE y-coordinates:{MSE_y}')
    print(f'MSE z-coordinates:{MSE_z}')
    print(f'MSE location:{MSE_locations}')

    plot_MED_error(error_locations, nyhead.cortex[:,plane], 'Euclidean Distance', numbr)


# print(np.mean(sulci_error))
# print(np.mean(gyri_error))
    # plot_MSE_error(error_x, nyhead.cortex[:,plane], 'x', numbr)
    # plot_MSE_error(error_y, nyhead.cortex[:,plane], 'y', numbr)
    # plot_MSE_error(error_z, nyhead.cortex[:,plane], 'z', numbr)




