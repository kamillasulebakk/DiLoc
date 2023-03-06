import numpy as np
import matplotlib.pyplot as plt
from lfpykit.eegmegcalc import NYHeadModel
import os
import h5py

plt.style.use('seaborn')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

big_data_path = '/Users/Kamilla/Documents/DiLoc-data'

def plot_MSE_NN(train_loss, test_loss, NN, act_func, batch_size, num_epochs, name = "NN"):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE) [mm]', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(train_loss), label='Train')
    ax.plot(np.log(test_loss), label='Test')
    ax.legend(fontsize=18)
    # fig.savefig(f'plots/{name}/MSE_{NN}_{act_func}_{batch_size}_{num_epochs}.png')
    fig.savefig(f'plots/finals/MSE_{NN}_{act_func}_{batch_size}_{num_epochs}.png')


def plot_MSE_CNN(train_loss, test_loss, NN, act_func, batch_size, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE) [mm]', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(train_loss), label='Train')
    ax.plot(np.log(test_loss), label='Test')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/07.feb/MSE_CNN_{NN}_{act_func}_{batch_size}_{num_epochs}.png')


def plot_interpolated_eeg_data(nyhead, eeg_i, x_pos, y_pos, eeg_new, x_new, y_new, i):
    fig = plt.figure()
    fig = plt.figure(figsize=[17, 7])

    ax_elecs = fig.add_subplot(1, 3, 1, aspect = 1)

    vmax = np.max(np.abs(eeg_i))
    v_range = vmax
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))

    electrode_measures = np.zeros((2, 231))
    for idx in range(len(eeg_i)):
        c = cmap(eeg_i[idx])
        electrode_measures[0][idx] = nyhead.elecs[0, idx]
        electrode_measures[1][idx] = nyhead.elecs[1, idx]

        ax_elecs.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c,
                 zorder=nyhead.elecs[2, idx])
    # ax_elecs.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=15, color='orange', zorder=1000)


    ax_eeg = fig.add_subplot(1, 3, 2, aspect = 1)
    ax_eeg_new = fig.add_subplot(1, 3, 3, aspect = 1)

    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6]) # This axis is just the colorbar

    vmax = np.max(np.abs(eeg_i))

    cmap = plt.cm.get_cmap('PRGn')
    vmap = lambda v: cmap((v + vmax) / (2*vmax))
    levels = np.linspace(-vmax, vmax, 60)

    contourf_kwargs = dict(levels=levels,
                           cmap="PRGn",
                           vmax=vmax,
                           vmin=-vmax,
                          extend="both")
    scatter_params = dict(cmap="bwr", vmin=-vmax, vmax=vmax, s=25)

    # Plot 3D location EEG electrodes
    img = ax_eeg.tricontourf(x_pos, y_pos, eeg_i, **contourf_kwargs)
    ax_eeg.tricontour(x_pos, y_pos, eeg_i, **contourf_kwargs)

    img = ax_eeg_new.tricontourf(x_new, y_new, eeg_new, **contourf_kwargs)
    ax_eeg_new.tricontour(x_new, y_new, eeg_new, **contourf_kwargs)

    cbar = plt.colorbar(img, cax=cax)
    cbar.ax.tick_params(labelsize=23)
    # cbar.set_label("µV", labelpad=-5, fontsize=30)
    cbar.set_ticks([-vmax, -vmax/2, 0, vmax/2, vmax])

    ax_elecs.tick_params(axis='both', which='major', labelsize=23)
    ax_eeg.tick_params(axis='both', which='major', labelsize=23)
    ax_eeg_new.tick_params(axis='both', which='major', labelsize=23)

    # ax_elecs.legend(fontsize=30)
    # ax_eeg.legend(fontsize=30)
    # ax_eeg_new.legend(fontsize=30)

    fig.savefig(f'plots/CNN/new_eeg_dipole_pos_{i}')
    plt.close(fig)


def plot_active_region(eeg, activity_center, activity_radius, active_idxs, numbr):

    num_active_vertexes = len(active_idxs)

    nyhead_file = os.path.join(big_data_path, "sa_nyhead.mat")
    head_data = h5py.File(nyhead_file, 'r')["sa"]
    cortex = np.array(head_data["cortex75K"]["vc"]) # Locations of every vertex in cortex
    elecs = np.array(head_data["locs_3D"]) # 3D locations of electrodes
    num_elecs = elecs.shape[1]

    fig = plt.figure(figsize=[12, 8])
    fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    ax1 = fig.add_subplot(231, aspect=1)
    ax1.set_xlabel("x (mm)", fontsize = 23.0)
    ax1.set_ylabel("z (mm)", fontsize = 23.0)
    ax2 = fig.add_subplot(232, aspect=1)
    ax2.set_xlabel("x (mm)", fontsize = 23.0)
    ax2.set_ylabel("y (mm)", fontsize = 23.0)
    ax3 = fig.add_subplot(233, aspect=1)
    ax3.set_xlabel("y (mm)", fontsize = 23.0)
    ax3.set_ylabel("z (mm)", fontsize = 23.0)
    ax4 = fig.add_subplot(234, aspect=1)
    ax4.set_xlabel("x (mm)", fontsize = 23.0)
    ax4.set_ylabel("z (mm)", fontsize = 23.0)
    ax5 = fig.add_subplot(235, aspect=1)
    ax5.set_xlabel("x (mm)", fontsize = 23.0)
    ax5.set_ylabel("y (mm)", fontsize = 23.0)
    ax6 = fig.add_subplot(236, aspect=1)
    ax6.set_xlabel("y (mm)", fontsize = 23.0)
    ax6.set_ylabel("z (mm)", fontsize = 23.0)
    cax = fig.add_axes([0.9, 0.55, 0.01, 0.3]) # This axis is just the colorbar

    eegmax = np.max(np.abs(eeg))
    scatter_params = dict(cmap="bwr", vmin=-eegmax, vmax=eegmax, s=50)

    # Plot 3D location EEG electrodes
    # Arrange point along different axes to avoid confusing overlapping points
    order = np.argsort(elecs[1, :])
    ax1.scatter(elecs[0, order], elecs[2, order], c=eeg[order], **scatter_params)
    order = np.argsort(elecs[2, :])
    ax2.scatter(elecs[0, order], elecs[1, order], c=eeg[order], **scatter_params)
    order = np.argsort(elecs[0, :])
    im = ax3.scatter(elecs[1, order], elecs[2, order], c=eeg[order], **scatter_params)

    # cbar = plt.colorbar(im, cax=cax, label="µV", size=23)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=23)

    # Plotting crossection of cortex around active region center
    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - activity_center[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(cortex[2, :] - activity_center[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(cortex[0, :] - activity_center[0]) < threshold)[0]

    ax4.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs], s=1, c='gray')
    ax5.scatter(cortex[0, xy_plane_idxs], cortex[1, xy_plane_idxs], s=1, c='gray')
    ax6.scatter(cortex[1, yz_plane_idxs], cortex[2, yz_plane_idxs], s=1, c='gray')

    # Mark active sources
    plot_idxs_xz = np.intersect1d(xz_plane_idxs, active_idxs)
    plot_idxs_xy = np.intersect1d(xy_plane_idxs, active_idxs)
    plot_idxs_yz = np.intersect1d(yz_plane_idxs, active_idxs)
    ax4.scatter(cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz], s=1, c='orange')
    ax5.scatter(cortex[0, plot_idxs_xy], cortex[1, plot_idxs_xy], s=1, c='orange')
    ax6.scatter(cortex[1, plot_idxs_yz], cortex[2, plot_idxs_yz], s=1, c='orange')

    # Plot outline of active region
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = activity_center[0] + activity_radius * np.cos(theta)
    circle_z = activity_center[2] + activity_radius * np.sin(theta)
    ax4.plot(circle_x, circle_z, ls='--', c='k')

    circle_x = activity_center[0] + activity_radius * np.cos(theta)
    circle_y = activity_center[1] + activity_radius * np.sin(theta)
    ax5.plot(circle_x, circle_y, ls='--', c='k')

    circle_y = activity_center[1] + activity_radius * np.cos(theta)
    circle_z = activity_center[2] + activity_radius * np.sin(theta)
    ax6.plot(circle_y, circle_z, ls='--', c='k')

    ax1.tick_params(axis='both', which='major', labelsize=23)
    ax2.tick_params(axis='both', which='major', labelsize=23)
    ax3.tick_params(axis='both', which='major', labelsize=23)
    ax4.tick_params(axis='both', which='major', labelsize=23)
    ax5.tick_params(axis='both', which='major', labelsize=23)
    ax6.tick_params(axis='both', which='major', labelsize=23)

    plt.savefig(f"plots/dipole_area/new_dipole_area_reduced_{numbr}.pdf")


def plot_dipoles(nyhead, name, eeg, dipole_pos_list, numbr):
    x_lim = [-100, 100]
    y_lim = [-130, 100]
    z_lim = [-160, 120]

    vmax = np.max(np.abs(eeg))
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))

    fig = plt.figure(figsize=[19, 10])

    ax1 = fig.add_subplot(131, aspect=1)
    ax1.set_xlabel("x (mm)", fontsize = 23.0)
    ax1.set_ylabel("y (mm)", fontsize = 23.0)
    ax2 = fig.add_subplot(132, aspect=1)
    ax2.set_xlabel("x (mm)", fontsize = 23.0)
    ax2.set_ylabel("z (mm)", fontsize = 23.0)
    ax3 = fig.add_subplot(133, aspect=1)
    ax3.set_xlabel("y (mm)", fontsize = 23.0)
    ax3.set_ylabel("z (mm)", fontsize = 23.0)

    electrode_measures = np.zeros((2, 231))
    for idx in range(eeg.shape[0]):
        c = cmap(eeg[idx])
        electrode_measures[0][idx] = nyhead.elecs[0, idx]
        electrode_measures[1][idx] = nyhead.elecs[1, idx]

        ax1.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c,
                 zorder=nyhead.elecs[2, idx])
        ax2.plot(nyhead.elecs[0, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c,
                 zorder=nyhead.elecs[1, idx])
        ax3.plot(nyhead.elecs[1, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c,
                 zorder=-nyhead.elecs[0, idx])


    if name == "multiple_dipoles":
        for i in range(len(dipole_pos_list)):
            ax1.plot(dipole_pos_list[i][0], dipole_pos_list[i][1], '*', ms=20, color='orange', zorder=1000)
            ax2.plot(dipole_pos_list[i][0], dipole_pos_list[i][2], '*', ms=20, color='orange', zorder=1000)
            ax3.plot(dipole_pos_list[i][1], dipole_pos_list[i][2], '*', ms=20, color='orange', zorder=1000)

    elif name == "single_dipole":
        ax1.plot(dipole_pos_list[0], dipole_pos_list[1], '*', ms=20, color='orange', zorder=1000)
        ax2.plot(dipole_pos_list[0], dipole_pos_list[2], '*', ms=20, color='orange', zorder=1000)
        ax3.plot(dipole_pos_list[1], dipole_pos_list[2], '*', ms=20, color='orange', zorder=1000)


    ax1.tick_params(axis='both', which='major', labelsize=23)
    ax2.tick_params(axis='both', which='major', labelsize=23)
    ax3.tick_params(axis='both', which='major', labelsize=23)

    if name == "multiple_dipoles":
        fig.savefig(f'plots/{name}/eeg_field_{len(dipole_pos_list)}_{numbr}.png')
        print(f'Finished producing figure {numbr}')

    elif name == "single_dipole":
        fig.savefig(f'plots/NN/eeg_field_noise_{numbr*100}.png')
        print(f'Finished producing figure with {numbr*100} % noise')

    plt.close(fig)

def plot_neighbour_dipoles(dipole_loc, neighbour_loc, dipole_eeg, neighbour_eeg, corex_loc, corex_loc_neighbour):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('Electrode number', fontsize=18)
    ax.set_ylabel('eeg [uV]', fontsize=18) #nanoampere micrometer

    ax.plot(dipole_eeg, label=f'Dipole located in {corex_loc} \
                                with coordinates [{dipole_loc[0]:.2f}, \
                                {dipole_loc[1]:.2f}, {dipole_loc[2]:.2f} ]')

    ax.plot(neighbour_eeg, label=f'Neighbouring dipole located in {corex_loc} \
                                with coordinates [{neighbour_loc[0]:.2f}, \
                                {neighbour_loc[1]:.2f}, {neighbour_loc[2]:.2f} ]')

    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize=16)

    # ax.legend(fontsize=18)
    fig.savefig(f'plots/neighbour_dipoles.png')