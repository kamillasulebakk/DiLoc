import os

import torch
from sklearn.model_selection import train_test_split    # type: ignore
import numpy as np

from utils import numpy_to_torch
from produce_data import return_interpolated_eeg_data


def determine_fname_prefix(determine_area: bool, determine_amplitude: bool):
    if determine_area:
        result = 'area'
    elif determine_amplitude:
        result = 'amplitudes'
    else:
        result = 'simple'
    return result


def generate_log_filename(parameters):
    result = determine_fname_prefix(
        parameters['determine_area'],
        parameters['determine_amplitude']
    )
    if parameters['custom_loss']:
        result += '_today_new_cnn'
        # result += '_today_tanh_all_the_way_custom_new_standarization'
    # result += f'_{parameters["hl_act_func"]}'
    result += f'_{parameters["batch_size"]}_{parameters["learning_rate"]}'
    result += f'_{parameters["momentum"]}_{parameters["weight_decay"]}'
    result += f'_{parameters["l1_lambda"]}_{parameters["N_epochs"]}'

    i = 0
    while os.path.isfile(os.path.join('results', result + f'_({i}).txt')):
        i += 1
    return result + f'_({i})'


def load_data_files(
    data_split: str,
    determine_area: bool,
    determine_amplitude: bool,
    N_samples: int,
    N_dipoles: int,
    interpolate: bool,
    ):
    name = determine_fname_prefix(determine_area, determine_amplitude)
    filename_base = f'data/{name}_{N_samples}_{N_dipoles}'
    # name = 'amplitudes'
    # filename_base = f'data/{name}_constA_{N_samples}_{N_dipoles}'
    if data_split == 'test':
        filename_suffix = 'test'
    else:
        filename_suffix = 'train-validation'
    eeg = np.load(filename_base + '_eeg_' + filename_suffix + '.npy')
    target = np.load(filename_base + '_targets_' + filename_suffix + '.npy')

    # new function
    if interpolate:
        print(f'You are now interpolating the EEG data with {N_dipoles} dipoles')
        eeg = return_interpolated_eeg_data(eeg)

    return eeg, target


def normalize(x, max_x, min_x):
    x_new = (x - min_x)/(max_x - min_x)
    return x_new


def denormalize(x, max_x, min_x):
    x_new = x * (max_x - min_x) + min_x
    return x_new


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_split, parameters):
        valid_data_splits = ['train', 'validation', 'test']
        if data_split not in valid_data_splits:
            raise ValueError(
                f'data_split must be in {valid_data_splits}, not {data_split}'
            )

        if parameters['N_dipoles'] > 1:
            if parameters['determine_area']:
                raise ValueError(
                    'determine_area should be False when N_dipoles > 1'
                )
            # if not parameters['determine_amplitude']:
            #     raise ValueError(
            #         'determine_amplitude should be True when N_dipoles > 1'
            #     )

        self.determine_area = parameters['determine_area']
        self.determine_amplitude = parameters['determine_amplitude']

        eeg, target = load_data_files(
            data_split,
            self.determine_area,
            self.determine_amplitude,
            parameters['N_samples'],
            parameters['N_dipoles'],
            parameters['interpolate']
        )


        # BIG RED NOTE IS THIS WRONG ???
        eeg = (eeg - np.mean(eeg, axis = 0))/np.std(eeg, axis = 0)
        # eeg = (eeg - np.mean(eeg))/np.std(eeg)



        self.max_targets = np.array([
            72.02555727958679,
            73.47751750051975,
            81.150386095047,
            10,
            15
        ])
        self.min_targets = np.array([
            -72.02555727958679,
            -106.12010800838469,
            -52.66008937358856,
            1,
            0
        ])
        if self.determine_area:
            self.min_targets[3] = 10/899

        target = self.normalize(target)

        eeg = numpy_to_torch(eeg)
        target = numpy_to_torch(target)

        if data_split == 'test':
            self._eeg, self._target = eeg, target
        else:
            self._eeg, self._target = self.split_data(eeg, target, data_split)

        if data_split != 'test':
            self.add_noise(parameters['noise_pct'])


    def normalize(self, target):
        if self.determine_area:
            for i in range(target.shape[1]):
                target[:, i] = normalize(target[:, i], self.max_targets[i], self.min_targets[i])
        elif self.determine_amplitude:
            for i in range(target.shape[1]):
                # indexed modulo 4 to work for multiple dipole sources
                target[:, i] = normalize(target[:, i], self.max_targets[i%4], self.min_targets[i%4])
        # elif target.shape[1] == 6:
        #     print('hello')
        #     for i in range(target.shape[1]):
        #         target[:, i] = normalize(target[:, i], self.max_targets[i%3], self.min_targets[i%3])

        return target

    def denormalize(self, target):
        if self.determine_area:
            for i in range(target.shape[1]):
                target[:, i] = denormalize(target[:, i], self.max_targets[i], self.min_targets[i])
        elif self.determine_amplitude:
            for i in range(target.shape[1]):
                # indexed modulo 4 to work for multiple dipole sources
                target[:, i] = denormalize(target[:, i], self.max_targets[i%4], self.min_targets[i%4])
        # elif target.shape[1] == 6:
        #     print('hello')
        #     for i in range(target.shape[1]):
        #         target[:, i] = denormalize(target[:, i], self.max_targets[i%3], self.min_targets[i%3])

        return target

    def add_noise(self, noise_pct):
        noise = torch.normal(0, torch.std(self._eeg) * noise_pct/100, size=self._eeg.shape)
        self._eeg += noise

    def split_data(self, eeg, target, train_test):
        eeg_train, eeg_val, target_train, target_val = train_test_split(
            eeg, target, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            return eeg_train, target_train
        return eeg_val, target_val

    def __getitem__(self, idx):
        eeg = self._eeg[idx]
        target = self._target[idx]
        return eeg, target

    def __len__(self):
        return self._eeg.shape[0]

    @property
    def eeg(self):
        return self._eeg.clone()

    @property
    def target(self):
        return self._target.clone()