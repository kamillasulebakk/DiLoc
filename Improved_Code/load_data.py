import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt
from scipy import interpolate
from plot import plot_dipoles, plot_interpolated_eeg_data, plot_active_region
import utils

def load_data(num_samples: int, name: str, shape: str = "1d", num_dipoles: int = 2):
    """
    Name is either "dipole_area" or "multiple_dipoles"
    Shape is either "1d", "2d" or "interpolated"
    """
    valid_names = ['dipole_area', 'multiple_dipoles']
    valid_shapes = ['1d', '2d', 'interpolated']
    if not name in valid_names:
        raise ValueError(f'name must be one of {valid_names}, not {name}')
    if not shape in valid_shapes:
        raise ValueError(f'shape must be one of {valid_shapes}, not {shape}')

    try:
        if name == "multiple_dipoles":
            eeg = np.load(f'data/new/{name}_eeg_{num_samples}_{num_dipoles}.npy')
            pos_list = np.load(f'data/new/{name}_locations_{num_samples}_{num_dipoles}.npy').T

        else:
            print(f"Loading data from file: data/{name}_eeg_{num_samples}_20mm.npy")
            eeg = np.load(f'data/new/{name}_eeg_{num_samples}_20mm.npy')
            # eeg = np.load(f'data/{name}_eeg_{num_samples}_20mm.npy')
            pos_list = np.load(f'data/new/{name}_locations_{num_samples}_20mm.npy').T


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


def load_electrode_positions():
    x_pos = np.load('data/electrode_positions_x.npy')
    y_pos = np.load('data/electrode_positions_y.npy')
    return x_pos, y_pos