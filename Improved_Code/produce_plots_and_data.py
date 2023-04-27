import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
from scipy import interpolate
from plot import plot_dipoles, plot_interpolated_eeg_data, plot_active_region
import utils

def return_multiple_dipoles(num_samples: int, num_dipoles: int, create_plot: bool = False):
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
    dipole_amplitudes = np.zeros((1, num_samples))

    for i in range(num_samples):
        eeg_i = np.zeros((1,231))
        dipole_pos_list = []
        A = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)

            eeg_i = calculate_eeg(nyhead, A).T

        eeg[i, :] = eeg_i
        # What do I mean by this? *num_dipoles?
        dipole_amplitudes[:,i] = A*num_dipoles

        if i < 5 and create_plot == True:
            plot_dipoles(nyhead, "multiple_dipoles", eeg[i], dipole_pos_list, i)

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
        dipole_amplitudes[:,i] = A_sum

        target = np.concatenate((dipole_locations_and_radii, dipole_amplitudes), axis=0)

        print(i)

        # if i < 6:
        #     plot_active_region(eeg[i], centers[:,i], radii[i], pos_idx, i)
        #     print(f'Finished producing figure {i}')

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
    p = np.array(([0.0], [0.0], [A])) * 1E7 # [nA* u m]
    # Rotates the direction of the dipole moment so that it is normal to the cerebral cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    # Generates the EEG signal that belongs to the dipole moment
    eeg_i = M @ p * 1E3 # [mV] -> uV unit conversion (eeg_i between 10 og 100)

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

    if name == 'multiple_dipoles':
        eeg, target = return_multiple_dipoles(num_samples, num_dipoles)
        np.save(f'data/test_{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/test_{name}_locations_{num_samples}_{num_dipoles}', target)
        name = f'multiple_dipoles_{num_dipoles}'

    elif name == 'dipole_area':
        eeg, target = return_dipole_area(num_samples)
        print(target)
        np.save(f'data/new/{name}_eeg_{num_samples}_20mm', eeg)
        np.save(f'data/new/{name}_locations_{num_samples}_20mm', target)

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
    x = dipole_location[0] + 1 # mm
    y = dipole_location[1] #+ 1 # mm
    z = dipole_location[2] # mm

    neighbour_idx = nyhead.return_closest_idx([x,y,z])
    neighbour_location = nyhead.cortex[:, neighbour_idx]

    nyhead.set_dipole_pos(neighbour_location)
    eeg_neighbour = calculate_eeg(nyhead)

    sulci_map_neighbour = sulci_map[neighbour_idx]
    normal_vec_neighbour = cortex_normals[:,neighbour_idx]

    if sulci_map_neighbour == 1:
        print('Neighbour dipole is located in sulcus')
        corex_loc_neighbour = 'sulcus'
    else:
        print('Neighbour dipole is located in gyrus')
        corex_loc_neighbour = 'gyrus'

    plot_neighbour_dipoles(dipole_location, neighbour_location,
                            eeg, eeg_neighbour, corex_loc, corex_loc_neighbour,
                            normal_vec, normal_vec_neighbour
                          )




if __name__ == '__main__':
    num_samples = 10_000
    # name = 'multiple_dipoles'
    # num_dipoles = 1
    # prepare_and_save_data(num_samples, name, num_dipoles)

    name = 'dipole_area'
    prepare_and_save_data(num_samples, name)