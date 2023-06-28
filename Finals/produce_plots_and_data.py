import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
from scipy import interpolate
from plot import plot_dipoles, plot_interpolated_eeg_data, plot_active_region, plot_normalized_population, plot_neighbour_dipoles
import utils
import os
import h5py
from matplotlib.widgets import Slider


big_data_path = '/Users/Kamilla/Documents/DiLoc-data'


def return_simple_dipole(num_samples:int, num_dipoles: int = 1):
    """
    Produce eeg data from single dipole moment for num_samples

    input:
        num_samples : int
            The number of samples/patients

        num_dipoles : int
            The number of desired diples

    returns:
        eeg : numpy array of floats, size (num_samples, 231)
            Electrical signals from single dipole in the cortex

        dipole_locations : numpy array of floats, size (3, num_samples)
            Position of single dipole for each sample
    """
    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=36)
    dipole_locations = rng.choice(nyhead.cortex, size=num_samples, axis=1) # [mm]

    eeg = np.zeros((num_samples, 231))

    for i in range(num_samples):
        nyhead.set_dipole_pos(dipole_locations[:,i])
        eeg_i = calculate_eeg(nyhead)
        eeg[i, :] = eeg_i.T


    # plotting the data without and with (different amount of) noise
    plot_dipoles(nyhead, "simple_dipole", eeg[0], dipole_locations[:,0], 0)

    noise_pcts = [0.1, 0.5, 1, 10, 50]
    for i, noise_pct in enumerate(noise_pcts):
        eeg_noise = eeg[0] + np.random.normal(0, np.std(eeg[0]) * noise_pct/100, eeg[0].shape) # add noice

        plot_dipoles(nyhead, "simple_dipole", eeg_noise, dipole_locations[:,0], noise_pct)

    return eeg, dipole_locations



def return_dipoles_w_amplitudes(num_samples: int, num_dipoles: int, create_plot: bool = True):
    """
    Produce eeg data from multiple dipole moments for num_samples
    input:
        num_samples : int
            The number of samples/patients
        num_dipoles : int
            The number of desired diples
    returns:
        eeg : numpy array of floats, size (num_samples, 231)
            Electrical signals from single dipole in the cortex
        dipole_locations : numpy array of floats, size (3, num_samples)
            Position of single dipole for each sample
    """
    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=36)
    dipole_locations = rng.choice(nyhead.cortex, size=num_samples*num_dipoles, axis=1) # [mm]

    eeg = np.zeros((num_samples, 231))
    dipole_amplitudes = np.zeros((1, num_samples*num_dipoles))

    for i in range(num_samples):
        eeg_i = np.zeros((1,231))
        dipole_pos_list = []
        A = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)

            dipole_amplitudes[:,j+(num_dipoles)*i] = A

            eeg_tmp = calculate_eeg(nyhead, A).T
            eeg_i += eeg_tmp

        eeg[i, :] = eeg_i

        if i < 5 and create_plot == True:
            plot_dipoles(nyhead, "dipoles_w_amplitudes", eeg[i], dipole_pos_list, i)

    target = np.concatenate((dipole_locations, dipole_amplitudes), axis=0)

    return eeg, target


def return_dipole_area(num_samples: int, radii_range: int = 20):
    """
    Produce eeg data from population of dipoles for num_samples
    and provides plots of the 10 first samples

    input:
        num_samples : int
            Number of samples/patients

        radii_range : int
            Largest desired radius for the dipole population [mm]
    returns:
    pos_idx : list with ints
        Indices of the positions in the brain within the given radii

    eeg : array with shape (num_samples, 231)
        Electrical signals from dipolepopulation in the cortex

    dipole_locations_and_radii : array with shape (4, num_samples)
        Center and radius of dipole population for each sample
    """
    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=36)
    # Center of dipoles
    centers = rng.choice(nyhead.cortex, size=num_samples, axis=1) # [mm]
    radii = rng.uniform(low=1, high=radii_range, size=num_samples) # [mm]

    eeg = np.zeros((num_samples, 231))
    dipole_locations_and_radii = np.zeros((4, num_samples))
    dipole_amplitudes = np.zeros((1, num_samples))


    for i in range(num_samples):
        print(centers[:,i])
        dipole_locations_and_radii[0,i] = centers[0,i]
        dipole_locations_and_radii[1,i] = centers[1,i]
        dipole_locations_and_radii[2,i] = centers[2,i]
        dipole_locations_and_radii[3,i] = radii[i]
        # pos index consist of multiple dipoles within a defined radius
        pos_idx = return_dipole_population(nyhead, centers[:,i], radii[i])

        A = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        A_sum = 0

        while len(pos_idx) < 1:
            radii[i] += 1
            pos_idx = return_dipole_population(nyhead, centers[:,i], radii[i])

        eeg_i = np.zeros((1,231))

        for idx in pos_idx:
            nyhead.set_dipole_pos(nyhead.cortex[:,idx])
            eeg_i += calculate_eeg(nyhead, A).T
            A_sum += A

        eeg[i, :] = eeg_i
        dipole_amplitudes[:,i] = A_sum/len(pos_idx)

        target = np.concatenate((dipole_locations_and_radii, dipole_amplitudes), axis=0)
        if i < 6:
            plot_active_region(eeg[i], centers[:,i], radii[i], pos_idx, i)
            print(f'Finished producing figure {i}')

    return eeg, target


def return_dipole_area_const_A(num_samples: int, radii_range: int = 20):
    """
    Produce eeg data from population of dipoles for num_samples
    and provides plots of the 10 first samples

    input:
        num_samples : int
            Number of samples/patients

        radii_range : int
            Largest desired radius for the dipole population [mm]
    returns:
    pos_idx : list with ints
        Indices of the positions in the brain within the given radii

    eeg : array with shape (num_samples, 231)
        Electrical signals from dipolepopulation in the cortex

    dipole_locations_and_radii : array with shape (4, num_samples)
        Center and radius of dipole population for each sample
    """
    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=136)
    # Center of dipoles
    centers = rng.choice(nyhead.cortex, size=num_samples, axis=1) # [mm]
    radii = rng.uniform(low=1, high=radii_range, size=num_samples) # [mm]

    eeg = np.zeros((num_samples, 231))
    dipole_locations_and_radii = np.zeros((4, num_samples))
    dipole_amplitudes = np.zeros((1, num_samples))

    A = 1 # every dipole has constant amplitude

    for i in range(num_samples):
        dipole_locations_and_radii[0,i] = centers[0,i]
        dipole_locations_and_radii[1,i] = centers[1,i]
        dipole_locations_and_radii[2,i] = centers[2,i]
        dipole_locations_and_radii[3,i] = radii[i]
        # pos index consist of multiple dipoles within a defined radius
        pos_idx = return_dipole_population(nyhead, centers[:,i], radii[i])

        A_sum = 0

        while len(pos_idx) < 1:
            radii[i] += 1
            pos_idx = return_dipole_population(nyhead, centers[:,i], radii[i])

        eeg_i = np.zeros((1,231))

        for idx in pos_idx:
            nyhead.set_dipole_pos(nyhead.cortex[:,idx])
            eeg_i += calculate_eeg(nyhead, A).T
            A_sum += A

        eeg[i, :] = eeg_i
        dipole_amplitudes[:,i] = A_sum

        target = np.concatenate((dipole_locations_and_radii, dipole_amplitudes), axis=0)
        # if i < 6:
        #     plot_active_region(eeg[i], centers[:,i], radii[i], pos_idx, i)
        #     print(f'Finished producing figure {i}')
        print(i)


    return eeg, target


def return_two_dipoles(num_samples: int, num_dipoles: int, create_plot: bool = True):
    """
    Produce eeg data from two dipole moments with a distance of min 30 cm
    """
    nyhead = NYHeadModel()

    rng = np.random.default_rng(seed=36)
    dipole_locations = rng.choice(nyhead.cortex, size=num_samples*num_dipoles, axis=1) # [mm]

    for i in range(len(dipole_locations)):
        pos = dipole_locations

        dist = np.sqrt((pos[0][i] - pos[0][i+1])**2 + (pos[1][i] - pos[1][i+1])**2 + (pos[2][i] - pos[2][i+1])**2)
        if dist > 20:
            print(pos[:,i], pos[:,i+1])

    input()

    eeg = np.zeros((num_samples, 231))
    dipole_amplitudes = np.zeros((1, num_samples*num_dipoles))

    for i in range(num_samples):
        eeg_i = np.zeros((1,231))
        dipole_pos_list = []
        A = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)

            dipole_amplitudes[:,j+(num_dipoles)*i] = A

            eeg_tmp = calculate_eeg(nyhead, A).T
            eeg_i += eeg_tmp

        eeg[i, :] = eeg_i

        if i < 5 and create_plot == True:
            plot_dipoles(nyhead, "return_dipoles_w_amplitudes", eeg[i], dipole_pos_list, i)

    target = np.concatenate((dipole_locations, dipole_amplitudes), axis=0)

    return eeg, target


def calculate_eeg(nyhead, A: int = 1.0):
    """
    Calculates the eeg signal from the dipole population

    returns:
        eeg_i : array of length (231)
            Combined eeg signal from the dipole population for a single patient
    """
    M = nyhead.get_transformation_matrix()
    # Dipole oriented in depth direction in the cortex
    p = np.array(([0.0], [0.0], [A])) * 1E7 # [nA* mu m]
    # Rotates the direction of the dipole moment so that it is normal to the cerebral cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    # Generates the EEG signal that belongs to the dipole moment
    eeg_i = M @ p * 1E3 # [mV] -> muV unit conversion (eeg_i between 10 og 100)

    return eeg_i


def return_dipole_population(nyhead, center, radius):
    """
    Find the indices of the positions in the brain within a given radius.
    input:
    center : array of length (3)
        [x, y, z] of the center of the dipole_population, given in mm
    radius: int
        Radius of dipole population, given in mm
    returns:
    pos_idx : list with ints
        Indices of the positions in the brain within the given radius
    """

    dist = np.sqrt((nyhead.cortex[0] - center[0])**2 + (nyhead.cortex[1] - center[1])**2 + (nyhead.cortex[2] - center[2])**2)
    pos_idx = np.where(dist < radius)[0]

    return pos_idx

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


def return_interpolated_eeg_data(eeg, num_samples, grid_shape : int = 20):
    nyhead = NYHeadModel()

    eeg_matrix = np.zeros((num_samples, grid_shape, grid_shape))
    x_pos, y_pos = load_electrode_positions()

    x0 = np.min(x_pos)
    x1 = np.max(x_pos)
    y0 = np.min(y_pos)
    y1 = np.max(y_pos)

    for i in range(num_samples):
        eeg_i = eeg[i,:]
        x_grid, y_grid = np.meshgrid(np.linspace(x0, x1, grid_shape), np.linspace(y0, y1, grid_shape))

        x_new = x_grid.flatten()
        y_new = y_grid.flatten()

        eeg_new = interpolate.griddata((x_pos, y_pos), eeg_i, (x_new, y_new), method='nearest')
        eeg_matrix[i, :, :] = np.reshape(eeg_new, (grid_shape,grid_shape))

        if i < 5:
            plot_interpolated_eeg_data(nyhead, eeg_i, x_pos, y_pos, eeg_new, x_new, y_new, i)

    return eeg_matrix


def return_2d_eeg_data(eeg, num_samples, grid_shape : int = 20):
    x_pos, y_pos = load_electrode_positions()

    x_indices = utils.indices_from_positions(x_pos, grid_shape - 1)
    y_indices = utils.indices_from_positions(y_pos, grid_shape - 1)

    eeg_matrix = np.zeros((num_samples, grid_shape, grid_shape))
    eeg_matrix[:, x_indices, y_indices] = eeg

    return eeg_matrix


def prepare_and_save_data(num_samples, name, num_dipoles : int = 1):

    if name == 'simple_dipole':
        eeg, target = return_simple_dipole(num_samples)
        np.save(f'data/{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/{name}_locations_{num_samples}_{num_dipoles}', target)

    elif name == 'dipoles_w_amplitudes':
        eeg, target = return_dipoles_w_amplitudes(num_samples, num_dipoles)
        np.save(f'data/{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/{name}_locations_{num_samples}_{num_dipoles}', target)

    elif name == 'dipole_area':
        # "new" refers to data with smaller amplitudes
        eeg, target = return_dipole_area(num_samples)
        np.save(f'data/new_A_{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/new_A_{name}_locations_{num_samples}_{num_dipoles}', target)

    elif name == 'dipole_area_const_A':
        eeg, target = return_dipole_area_const_A(num_samples)
        np.save(f'data/const_A_{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/const_A_{name}_locations_{num_samples}_{num_dipoles}', target)

    print(f'Finished producing data for {name} with {num_dipoles} dipole(s)')
    print(f'and writing mean and std to file.')


def save_electrode_positions():
    electrode_positions = np.array(nyhead.head_data["locs_2D"])
    x_pos = electrode_positions[1]
    y_pos = electrode_positions[4]
    np.save('data/electrode_positions_x', x_pos)
    np.save('data/electrode_positions_y', y_pos)


def find_neighbour_dipole():
    nyhead = NYHeadModel()
    sulci_map = np.array(nyhead.head_data["cortex75K"]["sulcimap"], dtype=int)[0]
    cortex_normals = np.array(nyhead.head_data["cortex75K"]["normals"])

    # import sys
    # import numpy
    # numpy.set_printoptions(threshold=sys.maxsize)
    #
    # print(np.where(np.array(sulci_map) == 1)[0])
    # input()

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
        corex_loc = 'sulcus'
    else:
        print('Dipole is located in gyrus')
        corex_loc = 'gyrus'

    # Doing the same for the neighbouring dipole
    x = dipole_location[0] + 45 # mm
    y = dipole_location[1] # mm
    z = dipole_location[2] # mm

    neighbour_idx = nyhead.return_closest_idx([x,y,z])

    # idx for dipole located in sulcus
    # neighbour_idx = 53550
    neighbour_location = nyhead.cortex[:, neighbour_idx]

    nyhead.set_dipole_pos(neighbour_location)
    eeg_neighbour = calculate_eeg(nyhead)

    sulci_map_neighbour = sulci_map[neighbour_idx]
    normal_vec_neighbour = cortex_normals[:,neighbour_idx]

    if sulci_map_neighbour == 1:
        print('Dipole is located in sulcus')
        corex_loc_neighbour = 'sulcus'
    else:
        print('Dipole is located in gyrus')
        corex_loc_neighbour = 'gyrus'

    plot_neighbour_dipoles(dipole_location, neighbour_location,
                            eeg, eeg_neighbour, corex_loc, corex_loc_neighbour,
                            normal_vec, normal_vec_neighbour
                          )



def split_data_set(eeg_filename, target_list_filename, N_dipoles):

    eeg = np.load(f'data/{eeg_filename}')
    target_list = np.load(f'data/{target_list_filename}').T

    N_samples = np.shape(eeg)[0]

    if np.shape(target_list)[1] != 3:
        if np.shape(target_list)[1] == 4:
            target_list = np.reshape(target_list, (N_samples, 4*N_dipoles))
        else:
            target_list = np.reshape(target_list, (N_samples, 5*N_dipoles))

    eeg_validate = eeg[:20000,:]
    pos_list_validate = target_list[:20000,:]

    eeg_train_test = eeg[20000:,:]
    pos_list_train_test = target_list[20000:,:]

    np.save(f'data/train_test_{eeg_filename}', eeg_train_test)
    np.save(f'data/train_test_{target_list_filename}', pos_list_train_test)

    np.save(f'data/validate_{eeg_filename}', eeg_validate)
    np.save(f'data/validate_{target_list_filename}', pos_list_validate)


if __name__ == '__main__':
    # num_samples = 70_000
    # name = 'dipoles_w_amplitudes'
    # num_dipoles = 1
    # prepare_and_save_data(num_samples, name, num_dipoles)
    # print('Finished')
    #
    # eeg_filename = 'dipoles_w_amplitudes_eeg_70000_1.npy'
    # pos_list_filename = 'dipoles_w_amplitudes_locations_70000_1.npy'
    # split_data_set(eeg_filename, pos_list_filename, 1)
    # print('Finished')
    #
    # num_samples = 70_000
    # name = 'dipole_area'
    # num_dipoles = 1
    # prepare_and_save_data(num_samples, name, num_dipoles)
    # print('Finished')

    # eeg_filename = 'new_A_dipole_area_eeg_70000_1.npy'
    # pos_list_filename = 'new_A_dipole_area_locations_70000_1.npy'
    # split_data_set(eeg_filename, pos_list_filename, 1)
    # print('Finished')

    # num_samples = 70_000
    # name = 'dipoles_w_amplitudes'
    # num_dipoles = 2
    # prepare_and_save_data(num_samples, name, num_dipoles)
    # print('Finished')

    # eeg_filename = 'dipoles_w_amplitudes_eeg_70000_2.npy'
    # pos_list_filename = 'dipoles_w_amplitudes_locations_70000_2.npy'
    # split_data_set(eeg_filename, pos_list_filename, 2)
    # print('Finished')

    # num_samples = 70_000
    # name = 'dipole_area_const_A'
    # num_dipoles = 1
    # prepare_and_save_data(num_samples, name, num_dipoles)
    # print('Finished')

    # eeg_filename = 'const_A_dipole_area_const_A_eeg_70000_1.npy'
    # pos_list_filename = 'const_A_dipole_area_const_A_locations_70000_1.npy'
    # split_data_set(eeg_filename, pos_list_filename, 1)
    # print('Finished')

    # plot_normalized_dipole_area(20)

    # return_dipole_area_const_A(10, 20)

    # radius = 5
    # plot_population_w_sliders(radius)

    # return_two_dipoles(10000, 2, True)

    # find_neighbour_dipole()

    return_simple_dipole(10, 1)