from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_simple_dipole import Net

from produce_plots_and_data import calculate_eeg
from utils import numpy_to_torch, normalize, denormalize, MSE, MAE
import produce_plots_and_data

import os
import h5py

def plot_MAE_error(mse, dipole_locs, name, numbr):
    fig = plt.figure(figsize=[8, 8])
    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    ax4 = fig.add_subplot(111, aspect=1, xlabel="x (mm)", ylabel="z (mm)")
    cax = fig.add_axes([0.92, 0.55, 0.01, 0.3]) # This axis is just the colorbar

    # mse_max = np.max(np.abs(mse))
    scatter_params = dict(cmap="hot", vmin=0, vmax=15, s=10)

    img = ax4.scatter(dipole_locs[0], dipole_locs[2], c=mse, **scatter_params)
    plt.colorbar(img, cax=cax)
    plt.savefig(f"plots/NEW_simple_dipole_error_{name}_{numbr}.pdf")


# x, y, z - coordinates
model = torch.load('trained_models/simple_dipole_lr0.001_RELU_500_50000.pt')

nyhead = NYHeadModel()

x_plane = 0
y_plane = 0
z_plane = 0

# Plotting crossection of cortex
threshold = 2  # threshold in mm for including points in plot

#inneholder alle idekser i koreks i planet mitt
xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - y_plane) < threshold)[0]
xy_plane_idxs = np.where(np.abs(nyhead.cortex[0, :] - z_plane) < threshold)[0]
yz_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] - x_plane) < threshold)[0]

planes = [xz_plane_idxs, xy_plane_idxs, yz_plane_idxs]

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

        error_i_locations = np.sqrt((error_i_x)**2 + (error_i_y)**2 + (error_i_z)**2)
        error_locations.append(error_i_locations)

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

    plot_MAE_error(error_x, nyhead.cortex[:,plane], 'x', numbr)
    plot_MAE_error(error_y, nyhead.cortex[:,plane], 'y', numbr)
    plot_MAE_error(error_z, nyhead.cortex[:,plane], 'z', numbr)
    plot_MAE_error(error_locations, nyhead.cortex[:,plane], 'Euclidean Distance', numbr)




