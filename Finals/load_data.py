import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import utils

def load_data_files(num_samples: int, name: str, shape: str = "1d", num_dipoles: int = 1):
    """
    Name is either "dipole_area", "dipoles_w_amplitudes" or "simple_dipole"
    Shape is either "1d", "2d" or "interpolated"
    """
    valid_names = ['dipole_area', 'dipoles_w_amplitudes', 'simple_dipole']
    valid_shapes = ['1d', '2d', 'interpolated']
    if not name in valid_names:
        raise ValueError(f'name must be one of {valid_names}, not {name}')
    if not shape in valid_shapes:
        raise ValueError(f'shape must be one of {valid_shapes}, not {shape}')

    try:
        eeg = np.load(f'data/train_test_{name}_eeg_70000_{num_dipoles}.npy')
        pos_list = np.load(f'data/train_test_{name}_locations_70000_{num_dipoles}.npy')

        # eeg = np.load(f'data/train_test_const_A_{name}_const_A_eeg_70000_{num_dipoles}.npy')
        # pos_list = np.load(f'data/train_test_const_A_{name}_const_A_locations_70000_{num_dipoles}.npy')

    except FileNotFoundError as e:
        print(f'The eeg data you seek (num_samples = {num_samples}, name = {name}, shape = {shape}) has not yet been produced.')
        raise e

    # Necessary in case of CNN
    if shape == "interpolated":
        print(f'You are now interpolating the EEG data with {num_dipoles} dipoles')
        eeg = return_interpolated_eeg_data(eeg, num_samples)
    elif shape == "2d":
        eeg = return_2d_eeg_data(eeg, num_samples)

    return eeg, pos_list
