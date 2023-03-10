import os

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split    # type: ignore
import numpy as np

from produce_and_load_eeg_data import load_data, load_mean_std
from plot import plot_MSE_NN
from utils import numpy_to_torch


class Net(nn.Module):
    def __init__(self, N_dipoles: int, determine_area: bool = False):
        self.determine_area = determine_area
        super().__init__()
        # self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(231, 180)
        self.fc2 = nn.Linear(180, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 16)
        if determine_area:
            self.fc5 = nn.Linear(16, 4*N_dipoles)
        else:
            self.fc5 = nn.Linear(16, 3*N_dipoles)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)

        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, determine_area: bool, N_samples: int, N_dipoles: int, noise_pct: int = 10):
        if train_test not in ['train', 'test']:
            raise ValueError(f'Unknown train_test value {train_test}')

        if determine_area:
            name = 'dipole_area'
        # elif N_dipoles > 1:
        #     name = 'multiple_dipoles'
        # else:
        #     name = 'single_dipole'
        else:
            name = 'multiple_dipoles'

        eeg, pos_list = load_data(N_samples, name, num_dipoles=N_dipoles)

        # TODO: clean up this normalization stuff
        # creating a string to pass information to a function instead of
        # passing the arguments directly is not useful
        mean, std_dev = load_mean_std(N_samples, f'{name}_{N_dipoles}')

        # normalize input data
        eeg = (eeg - mean)/std_dev
        eeg = numpy_to_torch(eeg)

        # TODO: move this to the generating function in
        # produce_and_load_eeg_data.py
        if N_dipoles > 1:
            pos_list = np.reshape(pos_list, (N_samples, 3*N_dipoles))

        if determine_area:
            # normalize target coordinates
            pos_list[:, :-1] = (pos_list[:, :-1] - np.mean(pos_list[:, :-1]))/np.std(pos_list[:, :-1])
            # normalize target radii
            # pos_list[:, -1] = (pos_list[:, -1] - np.mean(pos_list[:, -1]))/np.std(pos_list[:, -1])

        pos_list = numpy_to_torch(pos_list)
        self.eeg, self.pos_list = self.split_data(eeg, pos_list, train_test, noise_pct)

    def split_data(self, eeg, pos_list, train_test, noise_pct):
        eeg_train, eeg_test, pos_list_train, pos_list_test = train_test_split(
            eeg, pos_list, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            noise = torch.normal(0, torch.std(eeg_train) * noise_pct/100, size=eeg_train.shape)
            eeg_train += noise
            return eeg_train, pos_list_train
        if train_test == 'test':
            noise = torch.normal(0, torch.std(eeg_test) * noise_pct/10/100, size=eeg_test.shape)
            eeg_test += noise
            return eeg_test, pos_list_test

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        pos = self.pos_list[idx]
        return eeg, pos

    def __len__(self):
        return self.eeg.shape[0]


def train_epoch(data_loader_train, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, position) in enumerate(data_loader_train):
        optimizer.zero_grad()
        pred = net(signal)
        loss = criterion(pred, position)
        l1_lambda = 0.001

        # TODO: fix this list -> tensor hack
        l1_norm = torch.sum(torch.tensor([torch.linalg.norm(p, 1) for p in net.parameters()]))

        loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        losses[idx] = loss.item()
    mean_loss = np.mean(losses)
    return mean_loss


def test_epoch(data_loader_test, net, criterion):
    losses = np.zeros(len(data_loader_test))
    with torch.no_grad():
        for idx, (signal, position) in enumerate(data_loader_test):
            pred = net(signal)
            loss = criterion(pred, position)
            losses[idx] = loss.item()
        mean_loss = np.mean(losses)
    return mean_loss


def main(
    N_samples: int = 10_000,
    N_dipoles: int = 1,
    determine_area: bool = False,
    N_epochs: int = 2000,
    noise_pct: int = 10,
    log_dir: str = 'finals'
):
    msg = f'Training network with {N_samples} samples'
    if determine_area:
        msg += ' determining radii'
    else:
        msg += ' without determining radii'
    print(msg)
    print(f'{N_dipoles} dipole(s) and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 30

    net = Net(N_dipoles, determine_area)
    dataset_train = EEGDataset('train', determine_area, N_samples, N_dipoles)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    dataset_test = EEGDataset('test', determine_area, N_samples, N_dipoles)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6)

    # weight_decay > 0 --> l2/ridge penalty
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, momentum=1e-6, weight_decay = 1e-5)

    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    save_file_name = f'NN_{N_dipoles}_{N_samples}_l1_20mm'
    # save_file_name = f'NN_{N_dipoles}_{N_samples}_l1_l2'
    log_file_name = os.path.join(log_dir, save_file_name + '.txt')

    with open(log_file_name, 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'\nEeg Data: data/multiple_dipoles_eeg_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nLocation Data: data/multiple_dipoles_locations_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nMean and std Data: data/multiple_dipoles_{N_dipoles}_eeg_mean_std_{N_samples}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    status_line = 'Epoch {:4d}/{:4d} | Train: {:6.3f} | Test: {:6.3f} \n'
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(
            data_loader_train, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(
            data_loader_test, net, criterion)

        line = status_line.format(
            epoch, N_epochs - 1, train_loss[epoch], test_loss[epoch]
        )
        print(line)
        with open(log_file_name, 'a') as f:
            f.write(line)

        # print target and predicted values
        if epoch % 100 == 0:
            for i, (signal, position) in enumerate(data_loader_test):
                pred = net(signal)
                line = f'\n Target: {position[0]} \n'
                line += f'Predicted: {pred[0]} \n'
                print(line)
                with open(log_file_name, 'a') as f:
                    f.write(line)

                if i == 2:
                    with open(log_file_name, 'a') as f:
                        f.write('\n')
                    break

    # plot_MSE_NN(
    #     train_loss,
    #     test_loss,
    #     save_file_name,
    #     'TanH',
    #     batch_size,
    #     N_epochs,
    # )

    torch.save(net, f'trained_models/finals/{save_file_name}.pt')


    eeg, _ = load_data(N_samples, 'multiple_dipoles', num_dipoles=N_dipoles)
    input_names = ["EEG data"]
    output_names = ["Dipole source location"]
    torch.onnx.export(net, eeg.tolist(), "model.onnx", input_names=input_names, output_names=output_names)



if __name__ == '__main__':
    main(
        N_samples=10_000,
        N_dipoles=1,
        determine_area=False,
        N_epochs=3,
        noise_pct=10,
        log_dir='results/02.mar'
    )
