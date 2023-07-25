import os

import torch
from sklearn.model_selection import train_test_split    # type: ignore
import numpy as np

from utils import numpy_to_torch, normalize


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
    N_dipoles: int
    ):
    name = determine_fname_prefix(determine_area, determine_amplitude)
    filename_base = f'data/{name}_{N_samples}_{N_dipoles}'
    if data_split == 'test':
        filename_suffix = 'test'
    else:
        filename_suffix = 'train-validation'
    eeg = np.load(filename_base + '_eeg_' + filename_suffix + '.npy')
    target = np.load(filename_base + '_targets_' + filename_suffix + '.npy')
    return eeg, target


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
            if not parameters['determine_amplitude']:
                raise ValueError(
                    'determine_amplitude should be True when N_dipoles > 1'
                )

        eeg, target = load_data_files(
            data_split,
            parameters['determine_area'],
            parameters['determine_amplitude'],
            parameters['N_samples'],
            parameters['N_dipoles']
        )
        eeg = (eeg - np.mean(eeg))/np.std(eeg)

        max_targets = np.array([
            72.02555727958679,
            73.47751750051975,
            81.150386095047,
            10,
            15
        ])
        min_targets = np.array([
            -72.02555727958679,
            -106.12010800838469,
            -52.66008937358856,
            1,
            0
        ])

        if parameters['determine_area']:
            min_targets[3] = 10/899
            for i in range(target.shape[1]):
                target[:, i] = normalize(target[:, i], max_targets[i], min_targets[i])
        elif parameters['determine_amplitude']:
            for i in range(target.shape[1]):
                target[:, i] = normalize(target[:, i], max_targets[i], min_targets[i])

        eeg = numpy_to_torch(eeg)
        target = numpy_to_torch(target)

        if data_split == 'test':
            self.eeg, self.target = eeg, target
        else:
            self.eeg, self.target = self.split_data(eeg, target, data_split)
        self.add_noise(parameters['noise_pct'])

    def add_noise(self, noise_pct):
        noise = torch.normal(0, torch.std(self.eeg) * noise_pct/100, size=self.eeg.shape)
        self.eeg += noise

    def split_data(self, eeg, target, train_test):
        eeg_train, eeg_val, target_train, target_val = train_test_split(
            eeg, target, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            return eeg_train, target_train
        return eeg_val, target_val

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        target = self.target[idx]
        return eeg, target

    def __len__(self):
        return self.eeg.shape[0]
