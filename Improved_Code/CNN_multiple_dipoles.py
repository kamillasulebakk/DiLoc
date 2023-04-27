import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import produce_and_load_eeg_data
from plot import plot_MSE_CNN
import utils

from scipy import interpolate


class CNN(nn.Module):
    def __init__(self, N_dipoles):
        super().__init__()
        # Size of input image 20x20x1
        self.pool = nn.MaxPool2d(2, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.fc1 = nn.Linear(11*11*16, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 200)
        self.fc5 = nn.Linear(200, 3*N_dipoles)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    # def __init__(self, N_dipoles):
    #     super().__init__()
    #     # Size of input image 20x20x1
    #     self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    #     # Size: 16x16x6
    #     self.pool = nn.MaxPool2d(2, stride=1)
    #     # Size: 15x15x6
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     # Size: 11x11x16
    #     # Size after pool: 10x10x16
    #     self.fc1 = nn.Linear(10*10*16, 120)
    #     self.fc2 = nn.Linear(120, 64)
    #     self.fc3 = nn.Linear(64, 3*N_dipoles)
    #
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = torch.flatten(x, 1)     # flatten all dimensions except batch
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, name: str, N_samples: int, N_dipoles:int):

        mean, std_dev = produce_and_load_eeg_data.load_mean_std(N_samples, 'single_dipole')

        # Maybe 2d is not necesarry anymore
        # Use eeg data from selfdefined 2d matrix
        if name == '2d':
            eeg, pos_list = produce_and_load_eeg_data.load_data(N_samples, 'single_dipole', '2d')

        # Use eeg data from 2d interpolated data
        elif name == 'interpolated':
            eeg, pos_list = produce_and_load_eeg_data.load_data(N_samples, 'multiple_dipoles', 'interpolated', N_dipoles)

        # Scaling the data
        eeg = (eeg - mean)/std_dev

        print(N_dipoles)
        input()
        pos_list = np.reshape(pos_list, (N_samples, 3*N_dipoles))

        eeg_matrix = utils.numpy_to_torch(eeg)
        pos_list = utils.numpy_to_torch(pos_list)

        self.eeg_matrix, self.pos_list = self.split_data(eeg_matrix, pos_list, train_test)


    def split_data(self, eeg, pos_list, train_test):
        eeg_train, eeg_test, pos_list_train, pos_list_test = train_test_split(
            eeg, pos_list, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            return eeg_train, pos_list_train
        elif train_test == 'test':
            return eeg_test, pos_list_test
        else:
            raise ValueError(f'Unknown train_test value {train_test}')


    def __getitem__(self, idx):
        eeg = self.eeg_matrix[idx]
        pos = self.pos_list[idx]
        return eeg, pos


    def __len__(self):
        return self.eeg_matrix.shape[0]


def train_epoch(data_loader_train, noise_pct, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, position) in enumerate(data_loader_train):
        optimizer.zero_grad()
        noise = np.random.normal(0, np.std(signal.numpy()) * noise_pct/100, signal.shape)
        signal = utils.numpy_to_torch(signal + noise)
        signal = signal.unsqueeze(1)
        pred = net(signal)
        loss = criterion(pred, position)
        loss.backward()
        optimizer.step()
        losses[idx] = loss.item()
    mean_loss = np.mean(losses)
    return mean_loss


def test_epoch(data_loader_test, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_test))
    with torch.no_grad():
        for idx, (signal, position) in enumerate(data_loader_test):
            signal = signal.unsqueeze(1)
            pred = net(signal)
            loss = criterion(pred, position)
            losses[idx] = loss.item()
        mean_loss = np.mean(losses)
    return mean_loss


def main(name: str, N_samples = 10_000, N_dipoles = 2, N_epochs = 2000, noise_pct = 10):
    N_samples = 10_000
    N_dipoles = 2
    N_epochs = 1000
    noise_pct = 10

    print(f'You are now training the CNN network with {N_samples} samples,')
    print(f'{N_dipoles} dipole(s) and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 30

    net = CNN(N_dipoles)
    dataset_train = EEGDataset('train', name, N_samples, N_dipoles)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataset_test = EEGDataset('test', name, N_samples, N_dipoles)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6)
    # optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.999), eps=1e-08)

    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    with open(f'results/07.feb/restult_dipoles_{N_dipoles}_CNN.txt', 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'\nEeg Data: data/multiple_dipoles_eeg_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nLocation Data: data/multiple_dipoles_locations_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nMean and std Data: data/multiple_dipoles_{N_dipoles}_eeg_mean_std_{N_samples}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(data_loader_train, noise_pct, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(data_loader_test, optimizer, net, criterion)
        line = f'epoch {epoch:6d}, train loss: {train_loss[epoch]:9.3f}'
        line += f', test loss: {test_loss[epoch]:9.3f}'
        print(line)

        with open(f'results/07.feb/restult_dipoles_{N_dipoles}_CNN.txt', 'a') as f:
            f.write(f'epoch {epoch:2d}, train loss: {train_loss[epoch]:9.3f}')
            f.write(f', test loss: {test_loss[epoch]:9.3f} \n')

    plot_MSE_CNN(train_loss, test_loss, f'dipoles_{N_dipoles}_{name}_CNN_20x20_10000', 'ReLu_and_SGD', batch_size , N_epochs)

    return net


if __name__ == '__main__':
    N_samples = 10_000
    N_dipoles = 2
    N_epochs = 1000
    noise_pct = 10

    net = main('interpolated', N_samples, N_epochs, noise_pct)

    PATH = f'trained_models/CNN_interpolated_10000_{N_dipoles}.pt'
    torch.save(net, PATH)







