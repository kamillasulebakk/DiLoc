import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
from scipy import interpolate
from plot import plot_dipoles, plot_interpolated_eeg_data, plot_active_region
import utils


def return_single_dipole_data(num_samples:int):
    """
    Produce eeg data from single dipole moment for num_samples

    input:
        num_samples : int
            The number of samples/patients

        interpolation : bool
            "True" if one wants to interpolate eeg data into 2d matrix
            (for use in CCNs), "False" if not

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
    plot_dipoles(nyhead, "single_dipole", eeg[0], dipole_locations[:,0], 0)

    noise_pcts = [0.1, 0.5, 1, 10, 50]
    for i, noise_pct in enumerate(noise_pcts):

        mu, std = load_mean_std(num_samples, name)
        eeg_noise = eeg[0] + np.random.normal(0, std * noise_pct/100, eeg[0].shape) # add noice

        plot_dipoles(nyhead, "single_dipole", eeg_noise, dipole_locations[:,0], noise_pct)

    return eeg, dipole_locations


def return_dipole_area(num_samples: int, radii_range: int = 5):
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

    # TODO: Remeber make sure that a location cannot be picked twice??
    rng = np.random.default_rng(seed=36)
    # Center of dipoles
    centers = rng.choice(nyhead.cortex, size=num_samples, axis=1) # [mm]
    radii = rng.choice(radii_range, size=num_samples, axis=1) # [mm]

    eeg = np.zeros((num_samples, 231))
    dipole_locations_and_radii = np.zeros((4, num_samples))

    # TODO: vurdere å legge in feil, hvis pos_idx ... for å unngå rad 0
    for i in range(num_samples):
        dipole_locations_and_radii[0,i] = centers[0,i]
        dipole_locations_and_radii[1,i] = centers[1,i]
        dipole_locations_and_radii[2,i] = centers[2,i]
        dipole_locations_and_radii[3,i] = radii[i]
        # pos index consist of multiple dipoles within a defined radius
        pos_idx = return_dipole_population(nyhead, centers[:,i], radii[i])

        eeg_i = np.zeros((1,231))

        for idx in pos_idx:
            nyhead.set_dipole_pos(nyhead.cortex[:,idx])
            eeg_i += calculate_eeg(nyhead).T
        eeg[i, :] = eeg_i

        if i < 6:
            plot_active_region(eeg[i], centers[:,i], radii[i], pos_idx, i)
            print(f'Finished producing figure {i}')

    return eeg, dipole_locations_and_radii


def return_multiple_dipoles(num_samples: int, num_dipoles):
    """
    Produce eeg data from multiple dipole moments for num_samples

    input:
        num_samples : int
            The number of samples/patients

        interpolation : bool
            "True" if one wants to interpolate eeg data into 2d matrix
            (for use in CCNs), "False" if not

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

    for i in range(num_samples):
        eeg_i = np.zeros((1,231))
        dipole_pos_list = []
        for j in range(num_dipoles):
            nyhead.set_dipole_pos(dipole_locations[:,j+(num_dipoles)*i])
            dipole_pos_list.append(nyhead.dipole_pos)
            dip_orientation = np.random.choice([-1, 1])
            eeg_i += calculate_eeg(nyhead, dip_orientation).T
        eeg[i, :] = eeg_i

        if i < 5:
            plot_dipoles(nyhead, "multiple_dipoles", eeg[i], dipole_pos_list, i)


    return eeg, dipole_locations

def calculate_eeg(nyhead, dip_orientation: int = 1.0):
    """
    Calculates the eeg signal from the dipole population

    returns:
        eeg_i : array of length (231)
            Combined eeg signal from the dipole population for a single patient
    """
    M = nyhead.get_transformation_matrix()
    # Dipole oriented in depth direction in the cortex
    # p = np.array(([0.0], [0.0], [dip_orientation]))
    p = np.array(([0.0], [0.0], [dip_orientation])) * 1E7 # Ganske sterk dipol --> målbart resultat [nA* u m]
    # Rotates the direction of the dipole moment
    # so that it is normal to the cerebral cortex
    p = nyhead.rotate_dipole_to_surface_normal(p)
    # Generates the EEG signal that belongs to the dipole moment

    # eeg_i = M @ p * 1E9 # [mV] -> [pV] unit conversion
    eeg_i = M @ p * 1E3 # [mV] -> uV unit conversion (mellom 10 og 100)
    # HOW DOES pV affect MSE ?

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
    if name == 'single_dipole':
        eeg, dipole_info = return_single_dipole_data(num_samples)
        np.save(f'data/{name}_eeg_{num_samples}', eeg)
        np.save(f'data/{name}_locations_{num_samples}', dipole_info)

    elif name == 'dipole_area':
        eeg, dipole_info = return_dipole_area(num_samples)
        np.save(f'data/{name}_eeg_{num_samples}_5mm', eeg)
        np.save(f'data/{name}_locations_{num_samples}_5mm', dipole_info)

    elif name == 'multiple_dipoles':
        eeg, dipole_info = return_multiple_dipoles(num_samples, num_dipoles)
        np.save(f'data/{name}_eeg_{num_samples}_{num_dipoles}', eeg)
        np.save(f'data/{name}_locations_{num_samples}_{num_dipoles}', dipole_info)
        name = f'multiple_dipoles_{num_dipoles}'

    save_mean_std(eeg, name)
    print(f'Finished producing data for {name} with {num_dipoles} dipole(s)')
    print(f'and writing mean and std to file.')


def save_electrode_positions():
    electrode_positions = np.array(nyhead.head_data["locs_2D"])
    x_pos = electrode_positions[1]
    y_pos = electrode_positions[4]
    np.save('data/electrode_positions_x', x_pos)
    np.save('data/electrode_positions_y', y_pos)


def save_mean_std(eeg, name: str):
    num_samples = eeg.shape[0]
    eeg_mean = np.mean(eeg)
    eeg_std = np.std(eeg)
    with open(f'data/{name}_eeg_mean_std_{num_samples}_5mm.csv', 'w') as f:
        f.write(f'{eeg_mean},{eeg_std}')


# fix this. Is it necesarry to have all these if tests?
def load_data(num_samples: int, name: str, shape: str = "1d", num_dipoles: int = 2):
    """
    Name is either "single_dipole", "dipole_area" or "multiple_dipoles"
    Shape is either "1d", "2d" or "interpolated"
    """
    valid_names = ['single_dipole', 'dipole_area', 'multiple_dipoles']
    valid_shapes = ['1d', '2d', 'interpolated']
    if not name in valid_names:
        raise ValueError(f'name must be one of {valid_names}, not {name}')
    if not shape in valid_shapes:
        raise ValueError(f'shape must be one of {valid_shapes}, not {shape}')

    try:
        if name == "multiple_dipoles":
            eeg = np.load(f'data/{name}_eeg_{num_samples}_{num_dipoles}.npy')
            pos_list = np.load(f'data/{name}_locations_{num_samples}_{num_dipoles}.npy').T

        else:
            eeg = np.load(f'data/{name}_eeg_{num_samples}_5mm.npy')
            pos_list = np.load(f'data/{name}_locations_{num_samples}_5mm.npy').T


    except FileNotFoundError as e:
        print(f'The eeg data you seek (num_samples = {num_samples}, name = {name}, shape = {shape}) has not yet been produced.')
        raise e

    if shape == "interpolated":
        print(f'You are now interpolating the EEG data with {num_dipoles} dipoles')
        eeg = return_interpolated_eeg_data(eeg, num_samples)
    elif shape == "2d":
        eeg = return_2d_eeg_data(eeg, num_samples)


    return eeg, pos_list


def load_mean_std(num_samples, name: str):
    with open(f'data/{name}_eeg_mean_std_{num_samples}_5mm.csv', 'r') as f:
        mean, std = (float(e) for e in f.readline().split(','))

    filename = f'data/{name}_eeg_mean_std_{num_samples}_5mm.csv'
    return mean, std


def load_electrode_positions():
    x_pos = np.load('data/electrode_positions_x.npy')
    y_pos = np.load('data/electrode_positions_y.npy')
    return x_pos, y_pos


if __name__ == '__main__':
    num_samples = 10_000
    # name = 'multiple_dipoles'
    # num_dipoles = 2
    # prepare_and_save_data(num_samples, name, num_dipoles)

    # name = 'single_dipole'
    # prepare_and_save_data(num_samples, name, num_dipoles)

    name = 'dipole_area'
    prepare_and_save_data(num_samples, name)

