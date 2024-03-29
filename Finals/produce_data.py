import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
import utils
import os
import h5py
from matplotlib.widgets import Slider

from produce_plots import plot_dipoles, plot_interpolated_eeg_data, plot_active_region, plot_normalized_population, plot_neighbour_dipoles


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

    # noise_pcts = [0.1, 0.5, 1, 10, 50]
    noise_pcts = [10]

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
        A_tot = np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)
            A = A_tot/num_dipoles
            dipole_amplitudes[:,j+(num_dipoles)*i] = A

            eeg_tmp = calculate_eeg(nyhead, A).T
            eeg_i += eeg_tmp

        eeg[i, :] = eeg_i

        # if i < 5 and create_plot == True:
        #     plot_dipoles(nyhead, "dipoles_w_amplitudes", eeg[i], dipole_pos_list, i)

    target = np.concatenate((dipole_locations, dipole_amplitudes), axis=0)

    return eeg, target



def return_dipole_area_const_A(num_samples: int, radii_range: int = 15):
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
    dipole_centers = rng.choice(nyhead.cortex, size=num_samples, axis=1) # [mm]
    radii = rng.uniform(low=1, high=radii_range, size=num_samples) # [mm]

    eeg = np.zeros((num_samples, 231))
    dipole_amplitudes = np.zeros(num_samples)

    # max number of dipoles inside dipole population is 899
    # every dipole has constant amplitude
    A = 10/899 # max total amplitude for dipole population is 10

    for i in range(num_samples):
        pos_indices = return_dipole_population_indices(
            nyhead, dipole_centers[:,i], radii[i]
        )

        # ensure that population consists of at least one dipole
        while len(pos_indices) < 1:
            radii[i] += 1
            pos_idx = return_dipole_population_indices(nyhead, dipole_centers[:,i], radii[i])

        dipole_amplitudes[i] = A*len(pos_indices)
        for idx in pos_indices:
            nyhead.set_dipole_pos(nyhead.cortex[:, idx])
            eeg[i] += calculate_eeg(nyhead, A)

        if i < 6:
            plot_active_region(eeg[i], dipole_centers[:, i], radii[i], pos_indices, i)
            print(f'Finished producing figure {i}')


    target = np.concatenate(
        (dipole_centers, dipole_amplitudes.reshape(1, -1), radii.reshape(1, -1)),
        axis=0
    )

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

    eeg = np.zeros((num_samples, 231))
    dipole_amplitudes = np.zeros((1, num_samples*num_dipoles))

    for i in range(num_samples):
        eeg_i = np.zeros((1,231))
        dipole_pos_list = []
        A_tot = 2 #np.random.uniform(1, 10) # Amplitude. Each sample has its own amplitude value
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)

            A = A_tot/num_dipoles

            dipole_amplitudes[:,j+(num_dipoles)*i] = A

            eeg_tmp = calculate_eeg(nyhead, A).T
            eeg_i += eeg_tmp

        eeg[i, :] = eeg_i


        if i < 5 and create_plot == True:
            print(dipole_amplitudes[0,i*num_dipoles])
            plot_dipoles(nyhead, "dipoles_w_amplitudes", eeg[i], dipole_pos_list, i)

    target = dipole_locations
    # target = np.concatenate((dipole_locations, dipole_amplitudes), axis=0)

    return eeg, target


def calculate_eeg(nyhead, A: float = 1.0):
    """
    Calculates the eeg signal from the dipole population

    returns:
        eeg_i : array of length (231)
            Combined eeg signal from the dipole population for a single patient
    """
    M = nyhead.get_transformation_matrix()
    # Dipole oriented in depth direction in the cortex
    p = np.array(([0.0], [0.0], [A])) * 1E7 # [nA* mu m] => statisk dipol som representerer et valgt tidspunkt
    # Rotates the direction of the dipole moment so that it is normal to the cerebral cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    # Generates the EEG signal that belongs to the dipole moment
    eeg_i = M @ p * 1E3 # [mV] -> muV unit conversion (eeg_i between 10 og 100)

    return eeg_i.ravel()


def return_dipole_population_indices(nyhead, center, radius):
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
    # TODO: dist = np.linalg.norm(nyhead.cortex - center)
    pos_idx = np.where(dist < radius)[0]
    return pos_idx


def prepare_and_save_data(num_samples, name, num_dipoles : int = 1):

    if name == 'simple_dipole':
        eeg, target = return_simple_dipole(num_samples)
        np.save(f'data/simple_{num_samples}_{num_dipoles}_eeg_complete', eeg)
        np.save(f'data/simple_{num_samples}_{num_dipoles}_targets_complete', target)

    elif name == 'dipoles_w_amplitudes':
        if num_dipoles == 1:
            eeg, target = return_dipoles_w_amplitudes(num_samples, num_dipoles)
        else:
            eeg, target = return_two_dipoles(num_samples, num_dipoles)
        np.save(f'data/amplitudes_constA_{num_samples}_{num_dipoles}_eeg_complete', eeg)
        np.save(f'data/amplitudes_constA_{num_samples}_{num_dipoles}_targets_complete', target)

    elif name == 'dipole_area':
        eeg, target = return_dipole_area_const_A(num_samples)
        np.save(f'data/area_{num_samples}_{num_dipoles}_eeg_complete', eeg)
        np.save(f'data/area_{num_samples}_{num_dipoles}_targets_complete', target)

    print(f'Finished producing data for {name} with {num_dipoles} dipole(s)')
    print(f'and writing mean and std to file.')


def save_electrode_positions():
    electrode_positions = np.array(nyhead.head_data["locs_2D"])
    x_pos = electrode_positions[1]
    y_pos = electrode_positions[4]
    np.save('data/electrode_positions_x', x_pos)
    np.save('data/electrode_positions_y', y_pos)



def split_data_set(eeg_filename, target_list_filename, N_dipoles):

    eeg = np.load(f'data/{eeg_filename}')
    target_list = np.load(f'data/{target_list_filename}').T

    N_samples = np.shape(eeg)[0]

    if np.shape(target_list)[1] != 3:
        if np.shape(target_list)[1] == 4:
            target_list = np.reshape(target_list, (N_samples, 4*N_dipoles))
        else:
            target_list = np.reshape(target_list, (N_samples, 5*N_dipoles))


    if np.shape(target_list)[1] == 3 and N_dipoles>1:
        target_list = np.reshape(target_list, (N_samples, 3*N_dipoles))

    eeg_validate = eeg[:20000,:]
    pos_list_validate = target_list[:20000,:]

    eeg_train_test = eeg[20000:,:]
    pos_list_train_test = target_list[20000:,:]

    np.save(f'data/{eeg_filename}_train-validation', eeg_train_test)
    np.save(f'data/{target_list_filename}_train-validation', pos_list_train_test)

    np.save(f'data/{eeg_filename}_test', eeg_validate)
    np.save(f'data/{target_list_filename}_test', pos_list_validate)


if __name__ == '__main__':

    num_samples = 70_000
    # name = 'dipoles_w_amplitudes'
    num_dipoles = 2
    # prepare_and_save_data(num_samples, name, num_dipoles)
    return_two_dipoles(num_samples, num_dipoles, True)
    # return_dipole_area_const_A(num_samples)
    # return_simple_dipole(num_samples, 1)

    # eeg_filename = 'amplitudes_constA_70000_2_eeg_complete.npy'
    # pos_list_filename = 'amplitudes_constA_70000_2_targets_complete.npy'
    # split_data_set(eeg_filename, pos_list_filename, 2)
    # print('Finished')
