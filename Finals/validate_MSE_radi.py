from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

# from NN_simple_network_radius  import Net
from NN_dipole_area_working import Net


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

# model = torch.load('trained_models/july/new_dataset_simple_network_radius_tanh_sigmoid_50000_12july_MSEloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.1_bs32.pt')
model = torch.load('trained_models/50000_26junemseloss_MSE_area_w_amplitude_5000_SGD_lr0.001_wd0.1_mom0.35_bs64.pt')

eeg = np.load('data/validate_const_A_dipole_area_const_A_eeg_70000_1.npy')
target = np.load('data/validate_const_A_dipole_area_const_A_locations_70000_1.npy')
print('finished loading data')

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

# sulci_error_radius = []
# gyri_error_radius = []
#
# sulci_error_amplitude = []
# gyri_error_amplitude = []

for numbr, plane in enumerate(planes):
    error_locations = []
    error_x = []
    error_y = []
    error_z = []
    # error_radius = []
    # error_amplitude = []

    pred_list = np.zeros((len(plane),3))
    x_target_list = np.zeros(len(plane))
    y_target_list = np.zeros(len(plane))
    z_target_list = np.zeros(len(plane))
    # radius_target_list = np.zeros(len(plane))
    # amplitude_target_list = np.zeros(len(plane))


    for i, idx in enumerate(plane):
        nyhead.set_dipole_pos(nyhead.cortex[:,idx])
        eeg = calculate_eeg(nyhead)
        eeg = (eeg - np.mean(eeg))/np.std(eeg)
        eeg = numpy_to_torch(eeg.T)
        pred = model(eeg)
        pred = pred.detach().numpy().flatten()

        # radius_pred = pred_list[i, 3] = pred[3]
        # amplitude_pred = pred_list[i, 4] = pred[4]


        x_target = x_target_list[i] = nyhead.cortex[0,idx]
        y_target = y_target_list[i] = nyhead.cortex[1,idx]
        z_target = z_target_list[i] = nyhead.cortex[2,idx]

        x_pred =  pred_list[i, 0] = denormalize(pred[0], np.max(target[:,0]), np.min(target[:,0]))
        y_pred =  pred_list[i, 1] = denormalize(pred[1], np.max(target[:,1]), np.min(target[:,1]))
        z_pred =  pred_list[i, 2] = denormalize(pred[2], np.max(target[:,2]), np.min(target[:,2]))

        # radius_target = radius_target_list[i] = nyhead.cortex[3,idx]
        # amplitude_target = amplitude_target_list[i] = nyhead.cortex[4,idx]

        error_i_x = np.abs(x_target - x_pred)
        error_x.append(error_i_x)

        error_i_y = np.abs(y_target - y_pred)
        error_y.append(error_i_y)

        error_i_z = np.abs(z_target - z_pred)
        error_z.append(error_i_z)

        error_i_locations = (error_i_x + error_i_y + error_i_z)/3
        error_locations.append(error_i_locations)

        # error_i_radius = np.abs(radius_target - radius_pred)
        # error_radius.append(error_i_radius)
        #
        # error_i_amplitude = np.abs(amplitude_target - amplitude_pred)
        # error_amplitude.append(error_i_amplitude)

        if sulci_map[idx] == 1:
            sulci_error_location.append(error_i_locations)
            # sulci_error_radius.append(error_i_radius)
            # sulci_error_amplitude.append(error_i_amplitude)
        else:
            gyri_error_location.append(error_i_locations)
            # gyri_error_radius.append(error_i_radius)
            # gyri_error_amplitude.append(error_i_amplitude)


    target = np.concatenate((x_target_list, y_target_list, z_target_list))
    target = target.reshape((np.shape(pred_list)[1], np.shape(pred_list)[0]))
    target = target.T

    MSE_x = MSE(x_target_list, pred_list[:,0])
    MSE_y = MSE(y_target_list, pred_list[:,1])
    MSE_z = MSE(z_target_list, pred_list[:,2])

    MSE_locations = MSE(target, pred_list[:,:3])

    # MSE_radius = MSE(radius_target_list, pred_list[:,3])
    #
    # MSE_amplitude = MSE(amplitude_target_list, pred_list[:,4])

    print(f'MSE x-coordinates:{MSE_x}')
    print(f'MSE y-coordinates:{MSE_y}')
    print(f'MSE z-coordinates:{MSE_z}')
    print(f'MSE location:{MSE_locations}')
    print('')
    # print(f'MSE radius:{MSE_radius}')
    # print(f'MSE amplitude:{MSE_amplitude}')


print(np.mean(sulci_error_location))
print(np.mean(gyri_error_location))

    # plot_MSE_error(error_x, nyhead.cortex[:,plane], 'x', numbr)
    # plot_MSE_error(error_y, nyhead.cortex[:,plane], 'y', numbr)
    # plot_MSE_error(error_z, nyhead.cortex[:,plane], 'z', numbr)
    # plot_MSE_error(error_locations, nyhead.cortex[:,plane], 'Euclidean Distance', numbr)




