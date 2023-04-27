import os

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split    # type: ignore
import numpy as np

from load_data import load_data
from plot import plot_MSE_NN, plot_MSE_targets, plot_MSE_single_target
from utils import numpy_to_torch, normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau


class Net(nn.Module):
    def __init__(self, N_dipoles: int, determine_area: bool = False):
        self.determine_area = determine_area
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(231, 128*4)
        self.fc2 = nn.Linear(128*4, 64*4)
        self.fc3 = nn.Linear(64*4, 32*4)
        self.fc4 = nn.Linear(32*4, 16*4)
        self.fc5 = nn.Linear(16*4, 32)

        if determine_area:
            self.fc6 = nn.Linear(32, 5*N_dipoles)
        else:
            self.fc6 = nn.Linear(32, 4*N_dipoles)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)

        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, determine_area: bool, N_samples: int, N_dipoles: int, noise_pct: int = 10):
        if train_test not in ['train', 'test']:
            raise ValueError(f'Unknown train_test value {train_test}')

        if determine_area:
            name = 'dipole_area'
        else:
            name = 'multiple_dipoles'

        eeg, target = load_data(N_samples, name, num_dipoles=N_dipoles)


        # TODO: move this to the generating function in
        # produce_and_load_eeg_data.py
        if N_dipoles > 1:
            target = np.reshape(target, (N_samples, 3*N_dipoles))

        # normalize input data so that it ranges from 0 to 1
        # eeg = normalize(eeg)
        eeg = (eeg - np.mean(eeg))/np.std(eeg)


        if determine_area:
            # normalize target coordinates
            target[:, 0] = normalize(target[:, 0])
            target[:, 1] = normalize(target[:, 1])
            target[:, 2] = normalize(target[:, 2])
            # normalize target radii
            target[:, -2] = normalize(target[:, -1])
            # normalize target amplitude
            target[:, -1] = normalize(target[:, -1])

        else:
            # normalize target coordinates
            # normaliser x, y, z hver for seg

            target[:, 0] = normalize(target[:, 0])
            target[:, 1] = normalize(target[:, 1])
            target[:, 2] = normalize(target[:, 2])
            # normalize target amplitude
            target[:, -1] = normalize(target[:, -1])


        eeg = numpy_to_torch(eeg)
        target = numpy_to_torch(target)

        self.eeg, self.target = self.split_data(eeg, target, train_test, noise_pct)

    def split_data(self, eeg, target, train_test, noise_pct):
        eeg_train, eeg_test, target_train, target_test = train_test_split(
            eeg, target, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            noise = torch.normal(0, torch.std(eeg_train) * noise_pct/100, size=eeg_train.shape)
            eeg_train += noise
            return eeg_train, target_train
        if train_test == 'test':
            noise = torch.normal(0, torch.std(eeg_test) * noise_pct/100, size=eeg_test.shape)
            eeg_test += noise
            return eeg_test, target_test

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        target = self.target[idx]
        return eeg, target

    def __len__(self):
        return self.eeg.shape[0]


def train_epoch(data_loader_train, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, target_train) in enumerate(data_loader_train):
        optimizer.zero_grad()
        pred = net(signal)
        loss = criterion(pred, target_train)
        l1_lambda = 0.00001

        #TODO: fix this list -> tensor hack
        l1_norm = torch.sum(torch.tensor([torch.linalg.norm(p, 1) for p in net.parameters()]))

        loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        losses[idx] = loss.item()

    mean_loss = np.mean(losses)

    return mean_loss


def test_epoch(data_loader_test, net, criterion, scheduler):
    losses = np.zeros(len(data_loader_test))
    with torch.no_grad():
        for idx, (signal, target_test) in enumerate(data_loader_test):
            pred = net(signal)
            loss = criterion(pred, target_test)
            losses[idx] = loss.item()
        mean_loss = np.mean(losses)

        # Adjust the learning rate based on validation loss
        scheduler.step(losses[idx])

    return mean_loss


def main(
    N_samples: int = 10_000,
    N_dipoles: int = 1,
    determine_area: bool = False,
    N_epochs: int = 2000,
    noise_pct: int = 5,
    log_dir: str = 'results',
):
    noise_pct = noise_pct


    msg = f'Training network with {N_samples} samples'
    if determine_area:
        msg += ' determining radii'
    else:
        msg += ' without determining radii'
    print(msg)
    print(f'{N_dipoles} dipole(s) and {noise_pct} % noise for {N_epochs} epochs.\n')

    # batch_size = 30
    batch_size = 32


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

    lr = 1.8
    momentum = 1e-4
    weight_decay = 1e-5
    # weight_decay > 0 --> l2/ridge penalty

    save_file_name: str = f'new_lr_standarisation_w_amplitude_{N_epochs}_SGD_lr{lr}_wd{weight_decay}_bs{batch_size}'


    optimizer = torch.optim.SGD(net.parameters(), lr, momentum, weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=50,
                              verbose=True, threshold=0.0001, threshold_mode='rel',
                              cooldown=0, min_lr=0, eps=1e-08)

    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    MSE_x = np.zeros(N_epochs)
    MSE_y = np.zeros(N_epochs)
    MSE_z = np.zeros(N_epochs)
    MSE_A = np.zeros(N_epochs)

    log_file_name = os.path.join(log_dir, save_file_name + '.txt')

    with open(log_file_name, 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'Learning rate: {lr}, Momentum: {momentum}, Weight decay: {weight_decay} %\n \n')

        f.write(f'\nEeg Data: data/new/multiple_dipoles_eeg_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nTarget Data: data/new/multiple_dipoles_locations_{N_samples}_{N_dipoles}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    status_line = 'Epoch {:4d}/{:4d} | Train: {:6.6f} | Test: {:6.6f} \n'
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(
            data_loader_train, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(
            data_loader_test, net, criterion, scheduler)

        line = status_line.format(
            epoch, N_epochs - 1, train_loss[epoch], test_loss[epoch]
        )
        print(line)
        with open(log_file_name, 'a') as f:
            f.write(line)

        for i, (signal, target) in enumerate(data_loader_test):
            pred = net(signal)
            target_ = target.detach().numpy()
            pred_ = pred.detach().numpy()

            MSE_x[epoch] = np.mean((target_[:][0] - pred_[:][0]) ** 2)
            MSE_y[epoch] = np.mean((target_[:][1] - pred_[:][1]) ** 2)
            MSE_z[epoch] = np.mean((target_[:][2] - pred_[:][2]) ** 2)
            MSE_A[epoch] = np.mean((target_[:][3] - pred_[:][3]) ** 2)

        # print target and predicted values
        if epoch % 100 == 0:
            for i, (signal, target) in enumerate(data_loader_test):
                pred = net(signal)
                line = f'\n Target: {target[0]} \n'
                line += f'Predicted: {pred[0]} \n'
                print(line)
                with open(log_file_name, 'a') as f:
                    f.write(line)

                if i == 2:
                    with open(log_file_name, 'a') as f:
                        f.write('\n')
                    break

    plot_MSE_NN(
        train_loss,
        test_loss,
        save_file_name,
        'TanH',
        batch_size,
        N_epochs,
    )

    plot_MSE_targets(
        MSE_x,
        MSE_y,
        MSE_z,
        MSE_A,
        'TanH',
        batch_size,
        save_file_name
    )

    plot_MSE_single_target(
        MSE_x,
        'TanH',
        batch_size,
        save_file_name
    )

    torch.save(net, f'trained_models/{save_file_name}.pt')


if __name__ == '__main__':
    main(
        N_samples=10_000,
        N_dipoles=1,
        determine_area=False,
        N_epochs=500,
        noise_pct=10,
        log_dir='results'
    )