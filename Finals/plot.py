import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lfpykit.eegmegcalc import NYHeadModel
import os
# import h5py

plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)
# for e.g. \text command
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

big_data_path = '/Users/Kamilla/Documents/DiLoc-data'


def set_ax_info(ax, xlabel, ylabel, title=None, zlabel=None, legend=True):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        xlabel (str): the desired label on the x-axis
        ylabel (str): the desired label on the y-axis
        title (str, optional): the desired title on the axis
            default: None
        zlabel (str, optional): the desired label on the z-axis for 3D-plots
            default: None
        legend (bool, optional): whether or not to add labels/legend
            default: True
    """
    if zlabel == None:
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        # ax.ticklabel_format(style='plain')
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_zlabel(zlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.ticklabel_format(style='scientific', scilimits = (-2, 2))
    if title != None:
        ax.set_title(title, fontsize=20)
    if legend:
        ax.legend(fontsize=15)


def plot_MSE_targets_2_dipoles(
    MSE_x1,
    MSE_y1,
    MSE_z1,
    MSE_A1,
    MSE_x2,
    MSE_y2,
    MSE_z2,
    MSE_A2,
    act_func,
    batch_size,
    NN,
    N_dipoles
):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(MSE_x1), label='no 1, x position [mm]')
    ax.plot(np.log(MSE_y1), label='no 1, y position [mm]')
    ax.plot(np.log(MSE_z1), label='no 1, z position [mm]')
    ax.plot(np.log(MSE_A1), label='no 1, Amplitude [nA um]')
    ax.plot(np.log(MSE_x2), label='no 2, x position [mm]')
    ax.plot(np.log(MSE_y2), label='no 2, y position [mm]')
    ax.plot(np.log(MSE_z2), label='no 2, z position [mm]')
    ax.plot(np.log(MSE_A2), label='no 2, Amplitude [nA um]')
    ax.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig(f'plots/mse_targets_{NN}_2_dipoles.pdf')


def plot_MSE_targets(targets, batch_size, filename, N_dipoles):
    fig, ax = plt.subplots()
    labels = ['$x$ position [mm]', '$y$ position [mm]', '$z$ position [mm]', r'Amplitude [nA $\mu$m]']
    for target, label in zip(targets.T, labels):
        ax.plot(np.log(target), label=label)
    set_ax_info(
        ax,
        xlabel='Number of epochs',
        ylabel='ln(MSE)',
        title=f'MSE for validation data with batch size {batch_size}'
    )
    fig.tight_layout()
    fig.savefig(f'plots/mse_targets_{filename}.pdf')
    plt.close(fig)

def plot_MSE_single_target(target_1, act_func, batch_size, NN, N_dipoles):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(target_1, label='Amplitude [nA um]')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/mse_amplitude_{NN}.png')


def plot_MSE_NN(train_loss, test_loss, NN, act_func, batch_size, num_epochs, N_dipoles):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(train_loss), label='Train')
    ax.plot(np.log(test_loss), label='Test')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/MSE_{NN}_{act_func}_{batch_size}_{num_epochs}_N_dipoles_{N_dipoles}.png')

def plot_R2_NN(train_R2, test_R2, NN, act_func, batch_size, num_epochs, name = "NN"):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'R2 score for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    epoch_array = np.linspace(15, num_epochs, num_epochs-15)
    ax.plot(epoch_array, train_R2[15:], label='Train')
    ax.plot(epoch_array, test_R2[15:], label='Test')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/R2_{NN}_{act_func}_{batch_size}_{num_epochs}.png')


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

    plt.savefig(f"plots/dipole_area/dipole_area_reduced_{numbr}.pdf")


def plot_dipoles(nyhead, name, eeg, dipole_pos_list, numbr):
    sulci_map = np.array(nyhead.head_data["cortex75K"]["sulcimap"], dtype=int)[0]

    if len(dipole_pos_list) == 1:
        dipole_location = dipole_pos_list[0]
        idx = nyhead.return_closest_idx(dipole_location)
        sulci_map_dipole = sulci_map[idx]
        if sulci_map_dipole == 1:
            print(f'Dipole {numbr} is located in sulcus')
            corex_loc = 'sulcus'
        else:
            print(f'Dipole {numbr} is located in gyrus')
            corex_loc = 'gyrus'

    x_lim = [-100, 100]
    y_lim = [-130, 100]
    z_lim = [-160, 120]

    vmax = np.max(np.abs(eeg))
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))

    fig = plt.figure(figsize=[25, 11])
    # fig.suptitle(f'EEG signals from simulated current dipole moment(s)', fontsize=20)
    ax1 = fig.add_subplot(131, aspect=1)
    ax1.set_xlabel("x (mm)", fontsize = 30.0)
    ax1.set_ylabel("y (mm)", fontsize = 30.0)
    ax2 = fig.add_subplot(132, aspect=1)
    ax2.set_xlabel("x (mm)", fontsize = 30.0)
    ax2.set_ylabel("z (mm)", fontsize = 30.0)
    ax3 = fig.add_subplot(133, aspect=1)
    ax3.set_xlabel("y (mm)", fontsize = 30.0)
    ax3.set_ylabel("z (mm)", fontsize = 30.0)

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

    img = ax3.imshow([[], []], origin="lower", cmap=plt.cm.bwr)
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(label='µV',size=30, weight='bold')


    if name == "dipoles_w_amplitudes":
        for i in range(len(dipole_pos_list)):
            ax1.plot(dipole_pos_list[i][0], dipole_pos_list[i][1], '*', ms=22, color='orange', zorder=1000)
            ax2.plot(dipole_pos_list[i][0], dipole_pos_list[i][2], '*', ms=22, color='orange', zorder=1000)
            ax3.plot(dipole_pos_list[i][1], dipole_pos_list[i][2], '*', ms=22, color='orange', zorder=1000)

    elif name == "simple_dipole":
        ax1.plot(dipole_pos_list[0], dipole_pos_list[1], '*', ms=22, color='orange', zorder=1000)
        ax2.plot(dipole_pos_list[0], dipole_pos_list[2], '*', ms=22, color='orange', zorder=1000)
        ax3.plot(dipole_pos_list[1], dipole_pos_list[2], '*', ms=22, color='orange', zorder=1000)


    ax1.tick_params(axis='both', which='major', labelsize=27)
    ax2.tick_params(axis='both', which='major', labelsize=27)
    ax3.tick_params(axis='both', which='major', labelsize=27)

    if name == "dipoles_w_amplitudes":
        fig.savefig(f'plots/{name}_eeg_field_{len(dipole_pos_list)}_{numbr}.png')
        print(f'Finished producing figure {numbr}')

    elif name == "simple_dipole":
        fig.savefig(f'plots/{name}_eeg_field_noise_{numbr}.png')
        print(f'Finished producing figure with {numbr} % noise')

    plt.close(fig)

def plot_neighbour_dipoles(dipole_loc, neighbour_loc, dipole_eeg,
                            neighbour_eeg, corex_loc, corex_loc_neighbour,
                            normal_vec, normal_vec_neighbour
                            ):
    nyhead = NYHeadModel()
    head_colors = ["#ffb380", "#74abff", "#b3b3b3", "#c87137"]


    fig = plt.figure()

    x_lim = [-105, 105]
    y_lim = [-120, 110]
    z_lim = [-100, 110]

    ax_dict = dict(frameon=False, xticks=[], yticks=[], aspect=1)

    ax0_NY = fig.add_subplot(221, xlabel="x (mm)", ylabel='z (mm)',
                             xlim=x_lim, ylim=z_lim, **ax_dict)
    ax1_NY = fig.add_subplot(222, xlabel="x (mm)", ylabel='z (mm)',
                             xlim=x_lim, ylim=y_lim, **ax_dict)
    ax = fig.add_subplot(212, xlabel='Electrode number', ylabel='eeg [uV]')

    # print(normal_vec, normal_vec_neighbour)

    threshold = 1
    xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] -
                                dipole_loc[1]) < threshold)[0]
    cortex_x, cortex_z = nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs]
    ax0_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])
    ax1_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])


    p = np.array([0, 0, 1])

    nyhead.set_dipole_pos(dipole_loc)
    dipole_loc_arrow = nyhead.rotate_dipole_to_surface_normal(p)

    nyhead.set_dipole_pos(neighbour_loc)
    neighbour_loc_arrow = nyhead.rotate_dipole_to_surface_normal(p)

    arrow_plot_params = dict(lw=2, head_width=3, zorder=1000)

    dipole_arrow = dipole_loc_arrow / np.linalg.norm(dipole_loc_arrow) * 10
    ax0_NY.arrow(dipole_loc[0], dipole_loc[2],
                  dipole_arrow[0], dipole_arrow[2], color='blue', **arrow_plot_params,
                  label=f'Normal vector: ({normal_vec[0]:.2f}, \
                  {normal_vec[1]:.2f}, {normal_vec[2]:.2f} )')

    neighbour_arrow = neighbour_loc_arrow / np.linalg.norm(neighbour_loc_arrow) * 10
    ax1_NY.arrow(neighbour_loc[0], neighbour_loc[2],
                  neighbour_arrow[0], neighbour_arrow[2], color='green', **arrow_plot_params,
                  label=f'Normal vector: ({normal_vec_neighbour[0]:.2f}, \
                  {normal_vec_neighbour[1]:.2f}, {normal_vec_neighbour[2]:.2f} )')


    ax.plot(dipole_eeg, label=f'Blue dipole located in {corex_loc} \
                                with coordinates ({dipole_loc[0]:.2f}, \
                                {dipole_loc[1]:.2f}, {dipole_loc[2]:.2f} ) ')

    ax.plot(neighbour_eeg, label=f'Green dipole located in {corex_loc} \
                                with coordinates ({neighbour_loc[0]:.2f}, \
                                {neighbour_loc[1]:.2f}, {neighbour_loc[2]:.2f} )')

    ax0_NY.legend(fontsize=10, loc='upper left')
    ax1_NY.legend(fontsize=10, loc='upper right')
    ax.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', fontsize=10)

    fig.savefig(f'plots/compare_dipoles.png')


def plot_simple_example(A):
    nyhead = NYHeadModel()

    dipole_location = 'motorsensory_cortex'  # predefined location from NYHead class
    nyhead.set_dipole_pos(dipole_location)
    M = nyhead.get_transformation_matrix()

    p = np.array(([0.0], [0.0], [A])) * 1E7 # Ganske sterk dipol --> målbart resultat [nA* u m]

    # We rotate current dipole moment to be oriented along the normal vector of cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    eeg = M @ p * 1E9 # [mV] -> [pV] unit conversion

    x_lim = [-100, 100]
    y_lim = [-130, 100]
    z_lim = [-160, 120]

    input()

    plt.close("all")
    fig = plt.figure(figsize=[19, 10])
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17, wspace=0.3, left=0.1, right=0.99)
    ax1 = fig.add_subplot(245, aspect=1, xlabel="x (mm)", ylabel='y (mm)', xlim=x_lim, ylim=y_lim)
    ax2 = fig.add_subplot(246, aspect=1, xlabel="x (mm)", ylabel='z (mm)', xlim=x_lim, ylim=z_lim)
    ax3 = fig.add_subplot(247, aspect=1, xlabel="y (mm)", ylabel='z (mm)', xlim=y_lim, ylim=z_lim)
    # ax_eeg = fig.add_subplot(244, xlabel="Time (ms)", ylabel='pV', title='EEG at all electrodes')
    #
    # ax_cdm = fig.add_subplot(248, xlabel="Time (ms)", ylabel='nA$\cdot \mu$m',
    #                          title='Current dipole moment')
    # dist, closest_elec_idx = nyhead.find_closest_electrode()
    # print("Closest electrode to dipole: {:1.2f} mm".format(dist))

    max_elec_idx = np.argmax(np.std(eeg, axis=1))
    time_idx = np.argmax(np.abs(eeg[max_elec_idx]))
    max_eeg = np.max(np.abs(eeg[:, time_idx]))
    max_eeg_idx = np.argmax(np.abs(eeg[:, time_idx]))

    max_eeg_pos = nyhead.elecs[:3, max_eeg_idx]
    # fig.text(0.01, 0.25, "Cortex", va='center', rotation=90, fontsize=22)
    # fig.text(0.03, 0.25,
    #          "Dipole pos: {:1.1f}, {:1.1f}, {:1.1f}\nDipole moment: {:1.2f} {:1.2f} {:1.2f}".format(
    #     nyhead.dipole_pos[0], nyhead.dipole_pos[1], nyhead.dipole_pos[2],
    #     p[0, time_idx], p[1, time_idx], p[2, time_idx]
    # ), va='center', rotation=90, fontsize=14)

    # fig.text(0.01, 0.75, "EEG", va='center', rotation=90, fontsize=22)
    # fig.text(0.03, 0.75, "Max: {:1.2f} pV at idx {}\n({:1.1f}, {:1.1f} {:1.1f})".format(
    #          max_eeg, max_eeg_idx, max_eeg_pos[0], max_eeg_pos[1], max_eeg_pos[2]), va='center',
    #          rotation=90, fontsize=14)

    ax7 = fig.add_subplot(241, aspect=1, xlabel="x (mm)", ylabel='y (mm)',
                          xlim=x_lim, ylim=y_lim)
    ax8 = fig.add_subplot(242, aspect=1, xlabel="x (mm)", ylabel='z (mm)',
                          xlim=x_lim, ylim=z_lim)
    ax9 = fig.add_subplot(243, aspect=1, xlabel="y (mm)", ylabel='z (mm)',
                          xlim=y_lim, ylim=z_lim)

    # ax_cdm.plot(t, p[2, :], 'k')
    # [ax_eeg.plot(t, eeg[idx, :], c='gray') for idx in range(eeg.shape[0])]
    # ax_eeg.plot(t, eeg[closest_elec_idx, :], c='green', lw=2)

    vmax = np.max(np.abs(eeg[:, time_idx]))
    v_range = vmax
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2*vmax))

    threshold = 2

    xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] - nyhead.dipole_pos[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(nyhead.cortex[2, :] - nyhead.dipole_pos[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(nyhead.cortex[0, :] - nyhead.dipole_pos[0]) < threshold)[0]

    ax1.scatter(nyhead.cortex[0, xy_plane_idxs], nyhead.cortex[1, xy_plane_idxs], s=5)
    ax2.scatter(nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs], s=5)
    ax3.scatter(nyhead.cortex[1, yz_plane_idxs], nyhead.cortex[2, yz_plane_idxs], s=5)

    for idx in range(eeg.shape[0]):
        c = cmap(eeg[idx, time_idx])
        ax7.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c,
                 zorder=nyhead.elecs[2, idx])
        ax8.plot(nyhead.elecs[0, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c,
                 zorder=nyhead.elecs[1, idx])
        ax9.plot(nyhead.elecs[1, idx], nyhead.elecs[2, idx], 'o', ms=10, c=c,
                 zorder=-nyhead.elecs[0, idx])

    img = ax3.imshow([[], []], origin="lower", vmin=-vmax,
                     vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar(img, ax=ax9, shrink=0.5).set_label(label='µV',size=20, weight='bold')
    img.figure.axes[0].tick_params(axis="both", labelsize=20)
    img.figure.axes[1].tick_params(axis="x", labelsize=20)

        # img = ax3.imshow([[], []], origin="lower", cmap=plt.cm.bwr)
        # cb = plt.colorbar(img).


    ax1.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=12, color='orange', zorder=1000)
    ax2.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)
    ax3.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)

    ax7.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=12, color='orange', zorder=1000)
    ax8.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)
    ax9.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=12, color='orange', zorder=1000)

    plt.show()

def plot_normalized_population(eeg, activity_center, activity_radius, active_idxs):
    eeg = (eeg - np.mean(eeg))/np.std(eeg)
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

    plt.savefig(f"plots/dipole_area/large_dipole_area_normalized.pdf")

