from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_dipole_w_radi_amplitude import Net

from produce_plots_and_data import calculate_eeg
from utils import numpy_to_torch, normalize, denormalize, MSE, MAE, relative_change
import produce_plots_and_data

import os
import h5py

def plot_MAE_error(NN, mse, dipole_locs, name):
    fig = plt.figure(figsize=[8, 8])
    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    ax4 = fig.add_subplot(111, aspect=1, xlabel="x (mm)", ylabel="z (mm)")
    cax = fig.add_axes([0.92, 0.55, 0.01, 0.3]) # This axis is just the colorbar

    # mse_max = np.max(np.abs(mse))
    scatter_params = dict(cmap="hot", vmin=0, vmax=15, s=10)

    img = ax4.scatter(dipole_locs[0], dipole_locs[2], c=mse, **scatter_params)
    plt.colorbar(img, cax=cax)
    plt.savefig(f"plots/{name}_error_{name}.pdf")


# x, y, z - coordinates + radi + amplitude
model = torch.load('trained_models/dipole_w_radi_amplitude_500_SGD_lr1.5_wd0.1_mom0.35_bs64.pt')

nyhead = NYHeadModel()

error_locations = []
error_x = []
error_y = []
error_z = []

y_plane = 0

# Plotting crossection of cortex
threshold = 2  # threshold in mm for including points in plot

#inneholder alle idekser i koreks i planet mitt
xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - y_plane) < threshold)[0]

def find_max_min(coordinate):
    max = 0
    min = 0

    for idx in xz_plane_idxs:
        tmp = nyhead.cortex[coordinate,idx]

        if tmp > max:
            max = tmp
        elif tmp < min:
            min = tmp

    return max, min

pred_list = np.zeros((len(xz_plane_idxs),3))
x_target_list = np.zeros(len(xz_plane_idxs))
y_target_list = np.zeros(len(xz_plane_idxs))
z_target_list = np.zeros(len(xz_plane_idxs))

for i, idx in enumerate(xz_plane_idxs):
    nyhead.set_dipole_pos(nyhead.cortex[:,idx])
    eeg = calculate_eeg(nyhead)
    eeg = (eeg - np.mean(eeg))/np.std(eeg)
    eeg = numpy_to_torch(eeg.T)
    pred = model(eeg)
    pred = pred.detach().numpy().flatten()

    x_max, x_min = find_max_min(0)
    y_max, y_min = find_max_min(1)
    z_max, z_min = find_max_min(2)

    # denormalize target coordinates
    x_pred = pred_list[i, 0] = denormalize(pred[0], x_max, x_min)
    y_pred = pred_list[i, 1] = denormalize(pred[1], y_max, y_min)
    z_pred = pred_list[i, 2] = denormalize(pred[2], z_max, z_min)

    x_target = x_target_list[i] = nyhead.cortex[0,idx]
    y_target = y_target_list[i] = nyhead.cortex[1,idx]
    z_target = z_target_list[i] = nyhead.cortex[2,idx]

    # error_i_x = relative_change(x_target, x_pred)
    error_i_x = np.abs(x_target - x_pred)
    error_x.append(error_i_x)

    # error_i_y = relative_change(y_target, y_pred)
    error_i_y = np.abs(y_target - y_pred)
    error_y.append(error_i_y)

    # error_i_z = relative_change(z_target, z_pred)
    error_i_z = np.abs(z_target - z_pred)
    error_z.append(error_i_z)

    error_i_locations = np.sqrt((error_i_x)**2 + (error_i_y)**2 + (error_i_z)**2)
    error_locations.append(error_i_locations)

target = np.concatenate((x_target_list, y_target_list, z_target_list))

MAE_x = MAE(x_target_list, pred_list[:,0])
MAE_y = MAE(y_target_list, pred_list[:,1])
MAE_z = MAE(z_target_list, pred_list[:,2])
MAE_locations = (MAE_x + MAE_y + MAE_z) / 3

print(f'MAE x-coordinates:{MAE_x}')
print(f'MAE y-coordinates:{MAE_y}')
print(f'MAE z-coordinates:{MAE_z}')
print(f'MAE location:{MAE_locations}')

print(np.shape(error_x))
print(np.shape(nyhead.cortex[:,xz_plane_idxs]))
plot_MAE_error('dipole_w_amplitude', error_x, nyhead.cortex[:,xz_plane_idxs], 'x')
plot_MAE_error('dipole_w_amplitude', error_y, nyhead.cortex[:,xz_plane_idxs], 'y')
plot_MAE_error('dipole_w_amplitude', error_z, nyhead.cortex[:,xz_plane_idxs], 'z')
plot_MAE_error('dipole_w_amplitude', error_locations, nyhead.cortex[:,xz_plane_idxs], 'position')






