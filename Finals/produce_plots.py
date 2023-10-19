import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
from scipy import interpolate
import utils
import os
import h5py
import seaborn as sns
from matplotlib.widgets import Slider

from plot import set_ax_info
# from produce_data import calculate_eeg
big_data_path = '/Users/Kamilla/Documents/DiLoc-data'


plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)
palette = sns.color_palette("deep")
sns.set_palette(palette)

def plot_simple_example(A):
    nyhead = NYHeadModel()

    dipole_location = 'motorsensory_cortex'  # predefined location from NYHead class
    nyhead.set_dipole_pos(dipole_location)
    M = nyhead.get_transformation_matrix()

    p = np.array(([0.0], [0.0], [A])) * 1E7 # Ganske sterk dipol --> målbart resultat [nA* u m]

    # We rotate current dipole moment to be oriented along the normal vector of cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    eeg = M @ p * 1E3 # [mV] -> [pV] unit conversion

    eeg = eeg + np.random.normal(0, np.std(eeg) * 10/100, eeg.shape)

    x_lim = [-100, 100]
    y_lim = [-130, 100]
    z_lim = [-160, 120]

    fig = plt.figure(figsize=[19, 10])
    fig.subplots_adjust(top=0.96, bottom=0.05, hspace=0.17, wspace=0.3, left=0.1, right=0.99)
    ax1 = fig.add_subplot(245, aspect=1, xlabel=r"$x$ [mm]", ylabel=r'$y$ [mm]', xlim=x_lim, ylim=y_lim)
    ax2 = fig.add_subplot(246, aspect=1, xlabel=r"$x$ [mm]", ylabel=r'$z$ [mm]', xlim=x_lim, ylim=z_lim)
    ax3 = fig.add_subplot(247, aspect=1, xlabel=r"$y$ [mm]", ylabel=r'$z$ [mm]', xlim=y_lim, ylim=z_lim)


    max_elec_idx = np.argmax(np.std(eeg, axis=1))
    time_idx = np.argmax(np.abs(eeg[max_elec_idx]))
    max_eeg = np.max(np.abs(eeg[:, time_idx]))
    max_eeg_idx = np.argmax(np.abs(eeg[:, time_idx]))

    max_eeg_pos = nyhead.elecs[:3, max_eeg_idx]

    ax7 = fig.add_subplot(241, aspect=1, xlabel=r"$x$ [mm]", ylabel=r'$y$ [mm]',
                          xlim=x_lim, ylim=y_lim)
    ax8 = fig.add_subplot(242, aspect=1, xlabel=r"$x$ [mm]", ylabel=r'$z$ [mm]',
                          xlim=x_lim, ylim=z_lim)
    ax9 = fig.add_subplot(243, aspect=1, xlabel=r"$y$ [mm]", ylabel=r'$z$ [mm]',
                          xlim=y_lim, ylim=z_lim)

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


    ax1.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=15, color='orange', zorder=1000)
    ax2.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)
    ax3.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)

    ax7.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=15, color='orange', zorder=1000)
    ax8.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)
    ax9.plot(nyhead.dipole_pos[1], nyhead.dipole_pos[2], '*', ms=15, color='orange', zorder=1000)


    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=24)  # Set x-label font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=24)  # Set y-label font size
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)

    plt.tight_layout()
    plt.savefig(f"plots/simple_example.pdf")


def plot_different_amplitudes(A1, A2):
    nyhead = NYHeadModel()

    dipole_location = 'motorsensory_cortex'  # predefined location from NYHead class
    nyhead.set_dipole_pos(dipole_location)
    M = nyhead.get_transformation_matrix()

    p1 = np.array(([0.0], [0.0], [A1])) * 1E7  # Ganske sterk dipol --> målbart resultat [nA* u m]
    p2 = np.array(([0.0], [0.0], [A2])) * 1E7

    # We rotate current dipole moment to be oriented along the normal vector of cortex
    p1 = nyhead.rotate_dipole_to_surface_normal(p1)
    p2 = nyhead.rotate_dipole_to_surface_normal(p2)

    eeg1 = M @ p1 * 1E3  # [mV] -> [pV] unit conversion
    eeg2 = M @ p2 * 1E3

    x_lim = [-100, 100]
    y_lim = [-130, 100]

    plt.close("all")
    fig = plt.figure(figsize=[19, 10])
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    axes = [ax1, ax2]
    eeg_data = [eeg1, eeg2]
    titles = [f'Magnitude = {A1} nA$\mu$m', f'Magnitude = {A2} nA$\mu$m']

    vmax = np.max(np.abs(np.concatenate((eeg1, eeg2), axis=0)))
    cmap = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))

    for ax, eeg, title in zip(axes, eeg_data, titles):
        max_elec_idx = np.argmax(np.std(eeg, axis=1))
        time_idx = np.argmax(np.abs(eeg[max_elec_idx]))

        for idx in range(eeg.shape[0]):
            c = cmap(eeg[idx, time_idx])
            ax.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c,
                    zorder=nyhead.elecs[2, idx])

        ax.plot(nyhead.dipole_pos[0], nyhead.dipole_pos[1], '*', ms=20, color='orange', zorder=1000)
        ax.set_xlabel("x (mm)", fontsize=20)
        ax.set_ylabel("y (mm)", fontsize=20)
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(title, fontsize=25)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    img = ax1.imshow([[]], origin="lower", vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    cbar = plt.colorbar(img, cax=cbar_ax)
    cbar.set_label(label='µV', size=20, weight='bold')
    cbar.ax.tick_params(axis="both", labelsize=20)

    ax1.tick_params(axis='x', which='both', labelsize=20)
    ax1.tick_params(axis='y', which='both', labelsize=20)
    ax2.tick_params(axis='x', which='both', labelsize=20)
    ax2.tick_params(axis='y', which='both', labelsize=20)

    plt.savefig("plots/dipole_w_amplitude_example.pdf")


def plot_normalized_dipole_area(radius: float, random: bool):

    center = [-38.36417687, 3.95269847, 52.98116392]

    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=36)
    # center = rng.choice(nyhead.cortex) # [mm]

    eeg = np.zeros(231)
    dipole_location_and_radius = np.zeros(4)
    dipole_amplitude = 0

    dipole_location_and_radius[0] = center[0]
    dipole_location_and_radius[1] = center[1]
    dipole_location_and_radius[2] = center[2]
    dipole_location_and_radius[3] = radius
    # pos index consist of multiple dipoles within a defined radius
    pos_idx = return_dipole_population(nyhead, center[:], radius)

    if random:
        A = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
    else:
        A = 1

    A_sum = 0

    while len(pos_idx) < 1:
        radius += 1
        pos_idx = return_dipole_population(nyhead, center[:], radius)

    eeg_i = np.zeros((1,231))

    for idx in pos_idx:
        nyhead.set_dipole_pos(nyhead.cortex[:,idx])
        eeg_i += calculate_eeg(nyhead, A).T
        A_sum += A

    eeg[:] = eeg_i
    dipole_amplitude = A_sum/len(pos_idx)

    eeg = (eeg - np.mean(eeg))/np.std(eeg)

    return eeg, center, pos_idx
    # plot_normalized_population(eeg, center, radius, pos_idx)


def plot_population_w_sliders(initial_radius):
    # Create the figure and the line that we will manipulate
    eeg, center, active_idxs = plot_normalized_dipole_area(initial_radius, False)

    num_active_vertexes = len(active_idxs)

    nyhead_file = os.path.join(big_data_path, "sa_nyhead.mat")
    head_data = h5py.File(nyhead_file, 'r')["sa"]
    cortex = np.array(head_data["cortex75K"]["vc"])  # Locations of every vertex in cortex
    elecs = np.array(head_data["locs_3D"])  # 3D locations of electrodes
    num_elecs = elecs.shape[1]

    eegmax = np.max(np.abs(eeg))
    scatter_params = dict(cmap="bwr", vmin=-eegmax, vmax=eegmax, s=50)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    cax = fig.add_axes([0.9, 0.55, 0.01, 0.3])  # This axis is just the colorbar

    order = np.argsort(elecs[1, :])
    sc1 = ax1.scatter(elecs[0, order], elecs[2, order], c=eeg[order], **scatter_params)

    cbar = plt.colorbar(sc1, cax=cax)
    cbar.ax.tick_params(labelsize=23)

    # Plotting crossection of cortex around active region center
    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - center[1]) < threshold)[0]
    ax2.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs], s=1, c='gray')

    # Mark active sources
    plot_idxs_xz = np.intersect1d(xz_plane_idxs, active_idxs)
    sc2 = ax2.scatter(cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz], s=1, c='orange')

    # Plot outline of active region
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = center[0] + initial_radius * np.cos(theta)  # Use initial_radius here
    circle_z = center[2] + initial_radius * np.sin(theta)  # Use initial_radius here
    outline, = ax2.plot(circle_x, circle_z, ls='--', c='k')

    ax1.tick_params(axis='both', which='major', labelsize=23)
    ax2.tick_params(axis='both', which='major', labelsize=23)

    # Add activity radius slider
    slider_radius_ax = plt.axes([0.2, 0.02, 0.6, 0.03])  # Define the position of the radius slider
    slider_radius = Slider(slider_radius_ax, 'Radius', 0, 20, valinit=initial_radius, valstep=0.5)

    def update_radius(val):
        new_radius = slider_radius.val
        circle_x = center[0] + new_radius * np.cos(theta)  # Update circle_x based on new radius
        circle_z = center[2] + new_radius * np.sin(theta)  # Update circle_z based on new radius

        eeg_new, _, new_active_idxs = plot_normalized_dipole_area(new_radius, False)
        print(new_radius)
        print(eeg_new)

        sc1.set_array(eeg_new[order])  # Update the color mapping on the scatter plot

        outline.set_xdata(circle_x)  # Update the outline of active region
        outline.set_ydata(circle_z)
        plot_idxs_xz = np.intersect1d(xz_plane_idxs, new_active_idxs)
        sc2.set_offsets(np.column_stack((cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz])))  # Update the position of active sources

        fig.canvas.draw_idle()  # Redraw the figure

    slider_radius.on_changed(update_radius)  # Connect the slider to the update function

    plt.show()


def plot_and_find_neighbour_dipole():
    nyhead = NYHeadModel()
    sulci_map = np.array(nyhead.head_data["cortex75K"]["sulcimap"], dtype=int)[0]
    cortex_normals = np.array(nyhead.head_data["cortex75K"]["normals"])

    rng = np.random.default_rng(seed=36)

    dipole_location = rng.choice(nyhead.cortex, size=1, axis=1) # [mm]
    dipole_location = np.reshape(dipole_location, -1)

    nyhead.set_dipole_pos(dipole_location)
    eeg = calculate_eeg(nyhead)

    idx = nyhead.return_closest_idx(dipole_location)
    sulci_map_dipole = sulci_map[idx]
    normal_vec = cortex_normals[:,idx]

    if sulci_map_dipole == 1:
        print('Dipole is located in sulcus')
        cortex_loc = 'sulcus'
    else:
        print('Dipole is located in gyrus')
        cortex_loc = 'gyrus'

    # Doing the same for the neighbouring dipole
    # distance = -2

    x = dipole_location[0] # mm
    y = dipole_location[1] # mm
    z = dipole_location[2] # mm

    neighbour_idx_1 = nyhead.return_closest_idx([x-2,y,z])
    neighbour_idx_2 = nyhead.return_closest_idx([x+2,y,z])

    neighbour_idx = [neighbour_idx_1, neighbour_idx_2]
    # idx for dipole located in sulcus
    # neighbour_idx = 53550
    neighbour_locations = []
    eeg_neighbours = []
    sulci_map_neighbours = []
    normal_vec_neighbours = []
    cortex_loc_neighbours = []
    correlation_coefficients = []

    for i, neighbour in enumerate(neighbour_idx):
        neighbour_locations.append(nyhead.cortex[:, neighbour])
        nyhead.set_dipole_pos(neighbour_locations[i])
        eeg_neighbours.append(calculate_eeg(nyhead))

        sulci_map_neighbours.append(sulci_map[neighbour])
        normal_vec_neighbours.append(cortex_normals[:,neighbour])

        if sulci_map_neighbours == 1:
            print('Dipole is located in sulcus')
            cortex_loc_neighbours.append('sulcus')
        else:
            print('Dipole is located in gyrus')
            cortex_loc_neighbours.append('gyrus')

        correlation_coefficients.append(np.corrcoef(eeg, eeg_neighbours[i])[0, 1])
        print(f'The correlation coefficient between the EEG signals is {correlation_coefficients[i]}')

    plot_neighbour_dipoles(dipole_location, neighbour_locations,
                            eeg, eeg_neighbours, cortex_loc, cortex_loc_neighbours,
                            normal_vec, normal_vec_neighbours, correlation_coefficients
                          )


def plot_interpolated_eeg_data(eeg_i, x_pos, y_pos, eeg_new, x_new, y_new, i):
    nyhead = NYHeadModel()
    fig = plt.figure(figsize=[17, 7])

    ax_elecs = fig.add_subplot(1, 3, 1, aspect=1)

    vmax = np.max(np.abs(eeg_i))
    v_range = vmax

    electrode_measures = np.zeros((2, 231))
    for idx in range(len(eeg_i)):
        c = plt.cm.bwr((eeg_i[idx] + vmax) / (2 * vmax))  # Blue-White-Red colormap
        electrode_measures[0][idx] = nyhead.elecs[0, idx]
        electrode_measures[1][idx] = nyhead.elecs[1, idx]

        ax_elecs.plot(nyhead.elecs[0, idx], nyhead.elecs[1, idx], 'o', ms=10, c=c, zorder=nyhead.elecs[2, idx])

    ax_eeg = fig.add_subplot(1, 3, 2, aspect=1)
    ax_eeg_new = fig.add_subplot(1, 3, 3, aspect=1)

    cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])  # This axis is just the color bar

    vmax = np.max(np.abs(eeg_i))

    vmap = lambda v: plt.cm.bwr((v + vmax) / (2 * vmax))  # Blue-White-Red colormap
    levels = np.linspace(-vmax, vmax, 60)

    contourf_kwargs = dict(levels=levels, cmap="PRGn", vmax=vmax, vmin=-vmax, extend="both")  # Blue-White-Red colormap
    scatter_params = dict(cmap="PRGn", vmin=-vmax, vmax=vmax, s=25)  # Blue-White-Red colormap

    # Plot 3D location EEG electrodes
    img = ax_eeg.tricontourf(x_pos, y_pos, eeg_i, **contourf_kwargs)
    ax_eeg.tricontour(x_pos, y_pos, eeg_i, **contourf_kwargs)

    img = ax_eeg_new.tricontourf(x_new, y_new, eeg_new, **contourf_kwargs)
    ax_eeg_new.tricontour(x_new, y_new, eeg_new, **contourf_kwargs)

    cbar = plt.colorbar(img, cax=cax, cmap="PRGn")  # Specify the "bwr" colormap for the color bar
    cbar.ax.tick_params(labelsize=23)
    cbar.set_ticks([-vmax, -vmax / 2, 0, vmax / 2, vmax])

    ax_elecs.tick_params(axis='both', which='major', labelsize=23)
    ax_eeg.tick_params(axis='both', which='major', labelsize=23)
    ax_eeg_new.tick_params(axis='both', which='major', labelsize=23)

    fig.savefig(f'plots/CNN/one_dipole_eeg_dipole_pos_{i}')
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

    eeg = eeg.reshape(-1, 1)
    # print(np.max(eeg))
    # print(np.min(eeg))
    print(np.shape(eeg))

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

    max_elec_idx = np.argmax(np.std(eeg))
    time_idx = np.argmax(np.abs(eeg[max_elec_idx]))
    max_eeg = np.max(np.abs(eeg[:, time_idx]))
    max_eeg_idx = np.argmax(np.abs(eeg[:, time_idx]))

    max_eeg_pos = nyhead.elecs[:3, max_eeg_idx]

    vmax = np.max(np.abs(eeg[:, time_idx]))
    v_range = vmax
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

    img = ax3.imshow([[], []], origin="lower", vmin=-vmax,
                     vmax=vmax, cmap=plt.cm.bwr)
    cbar = plt.colorbar(img)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label(label='µV',size=30, weight='bold')



    # plt.colorbar(img, ax=ax9, shrink=0.5).set_label(label='µV',size=20, weight='bold')
    # img.figure.axes[0].tick_params(axis="both", labelsize=20)
    # img.figure.axes[1].tick_params(axis="x", labelsize=20)


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
        plt.tight_layout()
        fig.savefig(f'plots/{name}_eeg_field_{len(dipole_pos_list)}_{numbr}.pdf')
        print(f'Finished producing figure {numbr}')

    elif name == "simple_dipole":
        plt.tight_layout()
        fig.savefig(f'plots/{name}_eeg_field_noise_{numbr}.pdf')
        print(f'Finished producing figure with {numbr} % noise')

    plt.close(fig)

def plot_neighbour_dipoles(dipole_loc, neighbour_locs, dipole_eeg,
                            neighbour_eegs, cortex_loc, cortex_loc_neighbours,
                            normal_vec, normal_vec_neighbours, correlation_coefficients
                            ):
    nyhead = NYHeadModel()
    head_colors = ["#ffb380", "#74abff", "#b3b3b3", "#c87137"]


    fig = plt.figure()

    x_lim = [-105, 105]
    y_lim = [-120, 110]
    z_lim = [-100, 110]

    ax_dict = dict(frameon=False, xticks=[], yticks=[], aspect=1)

    ax1_NY = fig.add_subplot(331, xlabel="x (mm)", ylabel='z (mm)',
                                xlim=x_lim, ylim=y_lim, **ax_dict)
    ax0_NY = fig.add_subplot(332, xlim=x_lim, ylim=z_lim, **ax_dict)
    ax2_NY = fig.add_subplot(333, xlim=x_lim, ylim=y_lim, **ax_dict)

    ax0 = fig.add_subplot(312, ylabel=r'EEG [$\mu$V]')
    ax1 = fig.add_subplot(313, xlabel='Electrode number', ylabel=r'EEG [$\mu$V]')

    threshold = 1
    xz_plane_idxs = np.where(np.abs(nyhead.cortex[1, :] -
                                dipole_loc[1]) < threshold)[0]
    cortex_x, cortex_z = nyhead.cortex[0, xz_plane_idxs], nyhead.cortex[2, xz_plane_idxs]
    ax0_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])
    ax1_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])
    ax2_NY.scatter(cortex_x, cortex_z, s=4, c=head_colors[0])

    p = np.array([0, 0, 1])

    nyhead.set_dipole_pos(dipole_loc)
    dipole_loc_arrow = nyhead.rotate_dipole_to_surface_normal(p)

    nyhead.set_dipole_pos(neighbour_locs[0])
    neighbour_loc_arrow_1 = nyhead.rotate_dipole_to_surface_normal(p)

    nyhead.set_dipole_pos(neighbour_locs[1])
    neighbour_loc_arrow_2 = nyhead.rotate_dipole_to_surface_normal(p)

    arrow_plot_params = dict(lw=2, head_width=3, zorder=1000)

    dipole_arrow = dipole_loc_arrow / np.linalg.norm(dipole_loc_arrow) * 10
    ax0_NY.arrow(dipole_loc[0], dipole_loc[2],
                  dipole_arrow[0], dipole_arrow[2], color=palette[0], **arrow_plot_params,
                  label=r'$\newline \hat{\textbf{n}}$ = ' + f'({normal_vec[0]:.2f}, \
                  {normal_vec[1]:.2f}, {normal_vec[2]:.2f} )'
                  + r'\newline $\hat{\textbf{r}}$ = ' + f'({dipole_loc[0]:.2f}, \
                  {dipole_loc[1]:.2f}, {dipole_loc[2]:.2f} )')

    neighbour_arrow_1 = neighbour_loc_arrow_1 / np.linalg.norm(neighbour_loc_arrow_1) * 10
    neighbour_arrow_2 = neighbour_loc_arrow_2 / np.linalg.norm(neighbour_loc_arrow_2) * 10


    ax1_NY.arrow(neighbour_locs[0][0], neighbour_locs[0][2],
                  neighbour_arrow_1[0], neighbour_arrow_1[2], color=palette[2], **arrow_plot_params,
                  label=r'$\newline \hat{\textbf{n}}$ = ' + f'({normal_vec_neighbours[0][0]:.2f}, \
                  {normal_vec_neighbours[0][1]:.2f}, {normal_vec_neighbours[0][2]:.2f} )'
                  + r'\newline $\hat{\textbf{r}}$ = ' + f'({neighbour_locs[0][0]:.2f}, \
                  {neighbour_locs[0][1]:.2f}, {neighbour_locs[0][2]:.2f} )'
                  )

    ax2_NY.arrow(neighbour_locs[1][0], neighbour_locs[1][2],
                  neighbour_arrow_2[0], neighbour_arrow_2[2], color=palette[3], **arrow_plot_params,
                  label=r'$\newline \hat{\textbf{n}}$ = ' + f'({normal_vec_neighbours[1][0]:.2f}, \
                  {normal_vec_neighbours[1][1]:.2f}, {normal_vec_neighbours[1][2]:.2f} )'
                  + r'\newline $\hat{\textbf{r}}$ = ' + f'({neighbour_locs[1][0]:.2f}, \
                  {neighbour_locs[1][1]:.2f}, {neighbour_locs[1][2]:.2f})'
                  )


    ax0.plot(dipole_eeg, color=palette[0])

    ax0.plot(neighbour_eegs[0], color=palette[2])

    ax1.plot(dipole_eeg, color=palette[0])

    ax1.plot(neighbour_eegs[1], color=palette[3])

    ax0.set_yticks([-0.5, 0, 0.5, 1])  # Set ticks at the specified positions
    ax1.set_yticks([-0.5, 0, 0.5, 1])  # Set ticks at the specified positions

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)

    ax0.text(0.5, 0.9, f'Correlation coefficient: {correlation_coefficients[0]:.2f}', transform=ax0.transAxes,
         horizontalalignment='right', verticalalignment='top',
         fontsize=10, color='black', bbox=props)

    ax1.text(0.5, 0.9, f'Correlation coefficient: {correlation_coefficients[1]:.2f}', transform=ax1.transAxes,
         horizontalalignment='right', verticalalignment='top',
         fontsize=10, color='black', bbox=props)

    ax0_NY.legend(fontsize=10) #, loc='upper left')
    ax1_NY.legend(fontsize=10) #, loc='upper right')
    ax2_NY.legend(fontsize=10)


    plt.tight_layout()
    fig.savefig(f'plots/compare_dipoles.pdf')



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



def plot_prediction(name, taget, pred):
    nyhead = NYHeadModel()

    # target = [66.5, -26.5, 41.9]
    vertex_idx = nyhead.return_closest_idx(target)

    nyhead_file = os.path.join(big_data_path, "sa_nyhead.mat")
    head_data = h5py.File(nyhead_file, 'r')["sa"]
    cortex = np.array(head_data["cortex75K"]["vc"]) # Locations of every vertex in cortex
    elecs = np.array(head_data["locs_3D"]) # 3D locations of electrodes
    num_elecs = elecs.shape[1]

    # fig = plt.figure(figsize=[12, 8])
    # # fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    fig = plt.figure(figsize=[16, 8])  # Increase the figure width
    fig.subplots_adjust(hspace=0.6, wspace=-0.3, left=0.25, right=0.75, bottom=0.1, top=0.90)   # Adjust right margin


    ax1 = fig.add_subplot(321, aspect=1)
    ax1.set_xlabel("x (mm)", fontsize = 20.0)
    ax1.set_ylabel("z (mm)", fontsize = 20.0)
    ax2 = fig.add_subplot(323, aspect=1)
    ax2.set_xlabel("x (mm)", fontsize = 20.0)
    ax2.set_ylabel("y (mm)", fontsize = 20.0)
    ax3 = fig.add_subplot(325, aspect=1)
    ax3.set_xlabel("y (mm)", fontsize = 20.0)
    ax3.set_ylabel("z (mm)", fontsize = 20.0)

    ax4 = fig.add_subplot(322, aspect=1)
    ax4.set_xlabel("x (mm)", fontsize = 20.0)
    ax4.set_ylabel("z (mm)", fontsize = 20.0)
    ax5 = fig.add_subplot(324, aspect=1)
    ax5.set_xlabel("x (mm)", fontsize = 20.0)
    ax5.set_ylabel("y (mm)", fontsize = 20.0)
    ax6 = fig.add_subplot(326, aspect=1)
    ax6.set_xlabel("y (mm)", fontsize = 20.0)
    ax6.set_ylabel("z (mm)", fontsize = 20.0)


    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - target[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(cortex[2, :] - target[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(cortex[0, :] - target[0]) < threshold)[0]

    ax1.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs], s=1, c='gray')
    ax2.scatter(cortex[0, xy_plane_idxs], cortex[1, xy_plane_idxs], s=1, c='gray')
    ax3.scatter(cortex[1, yz_plane_idxs], cortex[2, yz_plane_idxs], s=1, c='gray')


    radius = 10
    # Plot outline of active region
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = target[0] + radius * np.cos(theta)
    circle_z = target[2] + radius * np.sin(theta)
    ax1.plot(circle_x, circle_z, ls='--', c='k')
    ax4.plot(circle_x, circle_z, ls='--', c='k')

    circle_x = target[0] + radius * np.cos(theta)
    circle_y = target[1] + radius * np.sin(theta)
    ax2.plot(circle_x, circle_y, ls='--', c='k')
    ax5.plot(circle_x, circle_y, ls='--', c='k')

    circle_y = target[1] + radius * np.cos(theta)
    circle_z = target[2] + radius * np.sin(theta)
    ax3.plot(circle_y, circle_z, ls='--', c='k')
    ax6.plot(circle_y, circle_z, ls='--', c='k')


    radius = 150
    dist = np.sqrt((nyhead.cortex[0] - target[0])**2 + (nyhead.cortex[1] - target[1])**2 + (nyhead.cortex[2] - target[2])**2)
    pos_idx = np.where(dist < radius)[0]

    # x_cortex = []
    # y_cortex = []
    # z_cortex = []


    threshold = 3  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - target[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(cortex[2, :] - target[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(cortex[0, :] - target[0]) < threshold)[0]

    # # Use the mask to scatter plot only the points within the circle
    # ax4.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs], s=1, c='gray')
    # ax5.scatter(cortex[0, xy_plane_idxs], cortex[1, xy_plane_idxs], s=1, c='gray')
    # ax6.scatter(cortex[1, yz_plane_idxs], cortex[2, yz_plane_idxs], s=1, c='gray')


    plot_idxs_xz = np.intersect1d(xz_plane_idxs, pos_idx)
    plot_idxs_xy = np.intersect1d(xy_plane_idxs, pos_idx)
    plot_idxs_yz = np.intersect1d(yz_plane_idxs, pos_idx)
    ax4.scatter(cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz], s=5, c='gray')
    ax5.scatter(cortex[0, plot_idxs_xy], cortex[1, plot_idxs_xy], s=5, c='gray')
    ax6.scatter(cortex[1, plot_idxs_yz], cortex[2, plot_idxs_yz], s=5, c='gray')

    ax4.scatter(cortex[0, vertex_idx], cortex[2, vertex_idx], s=20, c='orange')
    ax5.scatter(cortex[0, vertex_idx], cortex[1, vertex_idx], s=20, c='orange')
    ax6.scatter(cortex[1, vertex_idx], cortex[2, vertex_idx], s=20, c='orange')

    # pred = [66.4, -27.3, 41.5]
    ax4.scatter(66.4, 41.5, s=20, c='red')
    ax5.scatter(66.4, -27.3, s=20, c='red')
    ax6.scatter(-27.3, 41.5, s=20, c='red')


    ax1.set_xlim(-80, 80)
    ax2.set_xlim(-80, 80)
    ax3.set_xlim(-70, 20)
    ax4.set_xlim(50, 80)
    ax5.set_xlim(50, 80)
    ax6.set_xlim(-40, -10)

    ax1.set_ylim(-60, 100)
    ax2.set_ylim(-100, 70)
    ax3.set_ylim(-35, 55)
    ax4.set_ylim(30, 60)
    ax5.set_ylim(-45, -15)
    ax6.set_ylim(30, 60)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=20)
    ax6.tick_params(axis='both', which='major', labelsize=20)


    # ax_legend = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # Adjust the coordinates and size as needed
    ax_legend = fig.add_axes([0.75, 0.25, 0.02, 0.5])

    # Add red and orange dots to the legend axes
    ax_legend.scatter([0.2, 0.2], [0.6, 0.8], s=100, c=['red', 'orange'])

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis('off')  # Turn off the axis for the legend

    # Add text captions next to the dots
    ax_legend.text(0.5, 0.8, r'$\tilde{r}$ = [66.5, -26.5, 41.9]', fontsize=15, verticalalignment='center')
    ax_legend.text(0.5, 0.6, 'r = [66.4, -27.3, 41.5]', fontsize=15, verticalalignment='center')

    # ax4.legend(fontsize=15, loc='upper right')
    fig.suptitle('Example of FFNN Prediction for Location of Current Dipole', fontsize=20)
    # fig.subplots_adjust(hspace=0.6, wspace=0.1, left=0.07, right=0.9, bottom=0.1, top=0.95)

    plt.savefig(f"{name}_prediction.pdf")


def plot_prediction_multiple_dipoles(prediction, target, name):
    nyhead = NYHeadModel()

    target_1 = target[:3]
    target_2 = target[-3:]
    prediction_1 = prediction[:3]
    prediction_2 = prediction[-3:]

    vertex_idx_1 = nyhead.return_closest_idx(target_1)
    vertex_idx_2 = nyhead.return_closest_idx(target_2)

    vertex_list = [vertex_idx_1, vertex_idx_2]

    nyhead_file = os.path.join(big_data_path, "sa_nyhead.mat")
    head_data = h5py.File(nyhead_file, 'r')["sa"]
    cortex = np.array(head_data["cortex75K"]["vc"]) # Locations of every vertex in cortex
    elecs = np.array(head_data["locs_3D"]) # 3D locations of electrodes
    num_elecs = elecs.shape[1]

    # fig = plt.figure(figsize=[12, 8])
    # # fig.subplots_adjust(hspace=0.6, left=0.07, right=0.9, bottom=0.1, top=0.95)

    fig = plt.figure(figsize=[16, 8])  # Increase the figure width
    fig.subplots_adjust(hspace=0.6, wspace=0, left=0.25, right=0.75, bottom=0.1, top=0.90)   # Adjust right margin


    ax1 = fig.add_subplot(331, aspect=1)
    ax1.set_xlabel("x (mm)", fontsize = 20.0)
    ax1.set_ylabel("z (mm)", fontsize = 20.0)
    ax2 = fig.add_subplot(334, aspect=1)
    ax2.set_xlabel("x (mm)", fontsize = 20.0)
    ax2.set_ylabel("y (mm)", fontsize = 20.0)
    ax3 = fig.add_subplot(337, aspect=1)
    ax3.set_xlabel("y (mm)", fontsize = 20.0)
    ax3.set_ylabel("z (mm)", fontsize = 20.0)

    ax4 = fig.add_subplot(332, aspect=1)
    ax4.set_xlabel("x (mm)", fontsize = 20.0)
    ax4.set_ylabel("z (mm)", fontsize = 20.0)
    ax5 = fig.add_subplot(335, aspect=1)
    ax5.set_xlabel("x (mm)", fontsize = 20.0)
    ax5.set_ylabel("y (mm)", fontsize = 20.0)
    ax6 = fig.add_subplot(338, aspect=1)
    ax6.set_xlabel("y (mm)", fontsize = 20.0)
    ax6.set_ylabel("z (mm)", fontsize = 20.0)

    ax7 = fig.add_subplot(333, aspect=1)
    ax7.set_xlabel("x (mm)", fontsize = 20.0)
    ax7.set_ylabel("z (mm)", fontsize = 20.0)
    ax8 = fig.add_subplot(336, aspect=1)
    ax8.set_xlabel("x (mm)", fontsize = 20.0)
    ax8.set_ylabel("y (mm)", fontsize = 20.0)
    ax9 = fig.add_subplot(339, aspect=1)
    ax9.set_xlabel("y (mm)", fontsize = 20.0)
    ax9.set_ylabel("z (mm)", fontsize = 20.0)


    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - target_1[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(cortex[2, :] - target_1[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(cortex[0, :] - target_1[0]) < threshold)[0]


    ax1.scatter(cortex[0, xz_plane_idxs], cortex[2, xz_plane_idxs], s=1, c='gray')
    ax2.scatter(cortex[0, xy_plane_idxs], cortex[1, xy_plane_idxs], s=1, c='gray')
    ax3.scatter(cortex[1, yz_plane_idxs], cortex[2, yz_plane_idxs], s=1, c='gray')


    radius = 15
    # Plot outline of active region
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = target_1[0] + radius * np.cos(theta)
    circle_z = target_1[2] + radius * np.sin(theta)
    ax1.plot(circle_x, circle_z, ls='--', c='orange')
    ax4.plot(circle_x, circle_z, ls='--', c='orange')

    circle_x = target_1[0] + radius * np.cos(theta)
    circle_y = target_1[1] + radius * np.sin(theta)
    ax2.plot(circle_x, circle_y, ls='--', c='orange')
    ax5.plot(circle_x, circle_y, ls='--', c='orange')

    circle_y = target_1[1] + radius * np.cos(theta)
    circle_z = target_1[2] + radius * np.sin(theta)
    ax3.plot(circle_y, circle_z, ls='--', c='orange')
    ax6.plot(circle_y, circle_z, ls='--', c='orange')

    circle_x_2 = target_2[0] + radius * np.cos(theta)
    circle_z_2 = target_2[2] + radius * np.sin(theta)
    ax1.plot(circle_x_2, circle_z_2, ls='--', c='k')
    ax7.plot(circle_x_2, circle_z_2, ls='--', c='k')

    circle_x_2 = target_2[0] + radius * np.cos(theta)
    circle_y_2 = target_2[1] + radius * np.sin(theta)
    ax2.plot(circle_x_2, circle_y_2, ls='--', c='k')
    ax8.plot(circle_x_2, circle_y_2, ls='--', c='k')

    circle_y_2 = target_2[1] + radius * np.cos(theta)
    circle_z_2 = target_2[2] + radius * np.sin(theta)
    ax3.plot(circle_y_2, circle_z_2, ls='--', c='k')
    ax9.plot(circle_y_2, circle_z_2, ls='--', c='k')


    radius = 150
    dist = np.sqrt((nyhead.cortex[0] - target[0])**2 + (nyhead.cortex[1] - target[1])**2 + (nyhead.cortex[2] - target[2])**2)
    pos_idx = np.where(dist < radius)[0]

    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] - target[1]) < threshold)[0]
    xy_plane_idxs = np.where(np.abs(cortex[2, :] - target[2]) < threshold)[0]
    yz_plane_idxs = np.where(np.abs(cortex[0, :] - target[0]) < threshold)[0]


    plot_idxs_xz = np.intersect1d(xz_plane_idxs, pos_idx)
    plot_idxs_xy = np.intersect1d(xy_plane_idxs, pos_idx)
    plot_idxs_yz = np.intersect1d(yz_plane_idxs, pos_idx)

    ax4.scatter(cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz], s=5, c='gray')
    ax5.scatter(cortex[0, plot_idxs_xy], cortex[1, plot_idxs_xy], s=5, c='gray')
    ax6.scatter(cortex[1, plot_idxs_yz], cortex[2, plot_idxs_yz], s=5, c='gray')

    ax7.scatter(cortex[0, plot_idxs_xz], cortex[2, plot_idxs_xz], s=5, c='gray')
    ax8.scatter(cortex[0, plot_idxs_xy], cortex[1, plot_idxs_xy], s=5, c='gray')
    ax9.scatter(cortex[1, plot_idxs_yz], cortex[2, plot_idxs_yz], s=5, c='gray')

    ax4.scatter(cortex[0, vertex_idx_1], cortex[2, vertex_idx_1], s=20, c='orange')
    ax5.scatter(cortex[0, vertex_idx_1], cortex[1, vertex_idx_1], s=20, c='orange')
    ax6.scatter(cortex[1, vertex_idx_1], cortex[2, vertex_idx_1], s=20, c='orange')

    ax7.scatter(cortex[0, vertex_idx_2], cortex[2, vertex_idx_2], s=20, c='green')
    ax8.scatter(cortex[0, vertex_idx_2], cortex[1, vertex_idx_2], s=20, c='green')
    ax9.scatter(cortex[1, vertex_idx_2], cortex[2, vertex_idx_2], s=20, c='green')



    ax4.scatter(prediction_1[0], prediction_1[2], s=20, c='red')
    ax5.scatter(prediction_1[0], prediction_1[1], s=20, c='red')
    ax6.scatter(prediction_1[1], prediction_1[2], s=20, c='red')

    ax7.scatter(prediction_2[0], prediction_2[2], s=20, c='blue')
    ax8.scatter(prediction_2[0], prediction_2[1], s=20, c='blue')
    ax9.scatter(prediction_2[1], prediction_2[2], s=20, c='blue')

    ax1.set_xlim(-110, 110)
    ax2.set_xlim(-110, 110)
    ax3.set_xlim(-110, 110)
    ax4.set_xlim(25, 75)
    ax5.set_xlim(25, 75)
    ax6.set_xlim(-40, 10)
    ax7.set_xlim(-85, -35)
    ax8.set_xlim(-85, -35)
    ax9.set_xlim(-65, -15)

    ax1.set_ylim(-110, 110)
    ax2.set_ylim(-110, 110)
    ax3.set_ylim(-110, 110)
    ax4.set_ylim(-10, 40)
    ax5.set_ylim(-40, 10)
    ax6.set_ylim(-15, 35)
    ax7.set_ylim(0, 50)
    ax8.set_ylim(-65, -15)
    ax9.set_ylim(0, 50)

    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax3.tick_params(axis='both', which='major', labelsize=20)
    ax4.tick_params(axis='both', which='major', labelsize=20)
    ax5.tick_params(axis='both', which='major', labelsize=20)
    ax6.tick_params(axis='both', which='major', labelsize=20)
    ax7.tick_params(axis='both', which='major', labelsize=20)
    ax8.tick_params(axis='both', which='major', labelsize=20)
    ax9.tick_params(axis='both', which='major', labelsize=20)


    # ax_legend = fig.add_axes([0.91, 0.25, 0.02, 0.5])  # Adjust the coordinates and size as needed
    ax_legend = fig.add_axes([0.75, 0.25, 0.02, 0.5])

    # Add red and orange dots to the legend axes
    # ax_legend.scatter([0.1, 0.1], [0.6, 0.8], s=100, c=['red', 'orange'])
    ax_legend.scatter([0.2, 0.2, 0.2, 0.2], [0.8, 0.6, 0.4, 0.2], s=100, c=['red', 'orange', 'blue', 'green'])
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)
    ax_legend.axis('off')  # Turn off the axis for the legend

    # Add text captions next to the dots
    ax_legend.text(0.5, 0.6, r'$\tilde{\mathbf{r}_1}$ = (52.53, -16.25, 8.60)', fontsize=15, verticalalignment='center')
    ax_legend.text(0.5, 0.8, r'$\mathbf{r}_1$ = (55.95, -17.07, 13.43)', fontsize=15, verticalalignment='center')
    ax_legend.text(0.5, 0.2, r'$\tilde{\mathbf{r}_2}$ = (-63.48, -45.06, 22.67)', fontsize=15, verticalalignment='center')
    ax_legend.text(0.5, 0.4, r'$\mathbf{r}_2$ = (-64.09, -48.73, 19.89)', fontsize=15, verticalalignment='center')


    # ax4.legend(fontsize=15, loc='upper right')
    fig.suptitle('Example of Prediction for Location of Two Current Dipoles', fontsize=20)
    # fig.subplots_adjust(hspace=0.6, wspace=0.1, left=0.07, right=0.9, bottom=0.1, top=0.95)

    plt.savefig(f"{name}_prediction.pdf")


# plot_different_amplitudes(5, 10)
# plot_and_find_neighbour_dipole()
plot_simple_example(1)
# plot_prediction()


# This position is outside of the plot ... ?
# predicted = [-11.7499609,  -68.13421631,  59.55778122, -12.56062222, -88.71662903, 5.61093473]
# target = [-10.10664272, -61.96839905,  58.19534683, -10.3552618,  -87.21746063, 3.53685355]

# predicted = [55.9464798,  -17.07039642,  13.43354225, -64.09101868, -48.72733307, 19.88827515]
# target = [52.53390121, -16.25327682,   8.59553719, -63.4828872, -45.06331635, 22.66573524]
# name = 'two_dipoles'
#
# plot_prediction_multiple_dipoles(predicted, target, name)


# name = 'FFNN_single_dipole'
# pred = [66.9, -26.1, 41.7]
# target = [66.5, -26.4, 41.9]
# plot_prediction(name, target, pred)

# predicted = [-20.73835182  33.49762726  37.12372208 -37.32195282  11.24985504
#     62.58279037]
# target = [-12.03973293  34.7117424   60.07323456 -38.3641777    3.95269847
#     52.98116302]
#
# plot_prediction_multiple_dipoles(predicted, target)


# plot_simple_example(1)






















