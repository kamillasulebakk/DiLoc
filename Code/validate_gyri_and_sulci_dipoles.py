from lfpykit.eegmegcalc import NYHeadModel
import torch
import numpy as np
import matplotlib.pyplot as plt

from NN_17_feb import Net
from produce_and_load_eeg_data import calculate_eeg
from utils import numpy_to_torch
import produce_and_load_eeg_data

import os
import h5py

def plot_MSE_error(mse, dipole_locs):
    fig = plt.figure(figsize=[8, 8])
    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    ax4 = fig.add_subplot(111, aspect=1, xlabel="x (mm)", ylabel="z (mm)")
    cax = fig.add_axes([0.92, 0.55, 0.01, 0.3]) # This axis is just the colorbar

    mse_max = np.max(np.abs(mse))
    scatter_params = dict(cmap="hot", vmin=0, vmax=mse_max, s=10)

    img = ax4.scatter(dipole_locs[0], dipole_locs[2], c=mse, **scatter_params)
    plt.colorbar(img, cax=cax)
    plt.savefig(f"plots/mse_y_plane_test.pdf")


model = Net()
# model = torch.load('trained_models/NN_10000.pt')
model = torch.load('April/multipe_dipoles_lr0.0001_l1_11.april.pt')

nyhead = NYHeadModel()

# dipole_locations = [[37.8, -18.8, 71.1], [42.4, -18.8, 55.0]]
mse = []

# mean, std_dev = produce_and_load_eeg_data.load_mean_std(10_000)

sulcimap = np.array(nyhead.head_data["cortex75K"]["sulcimap"])[0,:]

idx_sulci = np.where(sulcimap == 1)
idx_sulci = idx_sulci[0]

idx_gyri = np.arange(0, len(nyhead.cortex[0]))
idx_gyri = np.delete(idx_gyri, idx_sulci)

y_plane = 0

# Plotting crossection of cortex
threshold = 2  # threshold in mm for including points in plot

#inneholder alle idekser i koreks i planet mitt
xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - y_plane) < threshold)[0]

for idx in xz_plane_idxs:
    nyhead.set_dipole_pos(nyhead.cortex[:,idx])
    eeg = calculate_eeg(nyhead)
    eeg = numpy_to_torch(eeg.T)
    eeg = (eeg - np.mean(eeg))/np.std(eeg)
    pred = model(eeg)

    mse_i = np.mean((nyhead.cortex[:,idx] - pred.detach().numpy())**2)
    mse.append(mse_i)

print(len(xz_plane_idxs))
plot_MSE_error(mse, nyhead.cortex[:,xz_plane_idxs])






