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
    def __init__(self):
        super().__init__()
        # Size of input image 20x20x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        # Size: 16x16x6
        self.pool = nn.MaxPool2d(2, stride=1)
        # Size: 15x15x6
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Size: 11x11x16
        # Size after pool: 10x10x16
        self.fc1 = nn.Linear(10*10*16, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_val: str, name: str, N_samples: int):

        mean, std_dev = produce_and_load_eeg_data.load_mean_std(N_samples, 'single_dipole')

        # Use eeg data from selfdefined 2d matrix
        if name == '2d':
            eeg, pos_list = produce_and_load_eeg_data.load_data(N_samples, 'single_dipole', '2d')

        # Use eeg data from 2d interpolated data
        elif name == 'interpolated':
            eeg, pos_list = produce_and_load_eeg_data.load_data(N_samples, 'single_dipole', 'interpolated')

        # Scaling the data
        eeg = (eeg - mean)/std_dev

        eeg_matrix = utils.numpy_to_torch(eeg)
        pos_list = utils.numpy_to_torch(pos_list)

        self.eeg_matrix, self.pos_list = self.split_data(eeg_matrix, pos_list, train_val)


    def split_data(self, eeg, pos_list, train_val):
        eeg_train, eeg_val, pos_list_train, pos_list_val = train_val_split(
            eeg, pos_list, val_size=0.2, random_state=0
        )
        if train_val == 'train':
            return eeg_train, pos_list_train
        elif train_val == 'val':
            return eeg_val, pos_list_val
        else:
            raise ValueError(f'Unknown train_val value {train_val}')


    def __getitem__(self, idx):
        eeg = self.eeg_matrix[idx]
        pos = self.pos_list[idx]
        return eeg, pos


    def __len__(self):
        return self.eeg_matrix.shape[0]


def train_epoch(data_loader_train, noise_pct, optimizer, net, criterion):
    total_loss = 0.0
    for eeg, target in data_loader:
        optimizer.zero_grad()
        noise = np.random.normal(0, np.std(eeg.numpy()) * noise_pct/100, signal.shape)
        eeg = utils.numpy_to_torch(eeg + noise)
        eeg = eeg.unsqueeze(1)
        pred = net(eeg)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss/len(data_loader)
    return mean_loss


def val_epoch(data_loader_val, optimizer, net, criterion):
    total_loss = 0.0    # NB! Will turn into torch.Tensor
    SE_targets = 0.0    # NB! Will turn into torch.Tensor
    total_number_of_samples = 0
    with torch.no_grad():
        for eeg, target in data_loader:
            signal = signal.unsqueeze(1)
            pred = net(eeg)
            loss = criterion(pred, target)
            total_loss += loss

            SE_targets += ((target - pred)**2).sum(dim=0)
            total_number_of_samples += target.shape[0]

        # Adjust the learning rate based on validation loss
        scheduler.step(total_loss)

    mean_loss = total_loss.item()/len(data_loader)
    MSE_targets = SE_targets.numpy()/total_number_of_samples

    return mean_loss, MSE_targets


def main(name: str, N_samples = 10_000, N_epochs = 2000, noise_pct = 10):
    print(f'You are now training the CNN with {N_samples} samples,')
    print(f'and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 32

    net = CNN()
    dataset_train = EEGDataset('train', name, N_samples)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataset_val = EEGDataset('val', name, N_samples)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                patience=50, verbose=True, threshold=0.0001,
                                threshold_mode='rel', cooldown=0, min_lr=0,
                                eps=1e-08)

    train_loss = np.zeros(N_epochs)
    val_loss = np.zeros(N_epochs)

    with open(f'results/01.feb/restult_single_dipole_CNN.txt', 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'\nEeg Data: data/single_dipole_eeg_{N_samples}.npy')
        f.write(f'\nLocation Data: data/single_dipole_locations_{N_samples}.npy')
        f.write(f'\nMean and std Data: data/single_dipole_eeg_mean_std_{N_samples}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(data_loader_train, noise_pct, optimizer, net, criterion)
        val_loss[epoch] = val_epoch(data_loader_val, optimizer, net, criterion)
        line = f'epoch {epoch:6d}, train loss: {train_loss[epoch]:9.3f}'
        line += f', val loss: {val_loss[epoch]:9.3f}'
        print(line)

        with open(f'results/01.feb/restult_single_dipole_CNN.txt', 'a') as f:
            f.write(f'epoch {epoch:2d}, train loss: {train_loss[epoch]:9.3f}')
            f.write(f', val loss: {val_loss[epoch]:9.3f} \n')

    plot_MSE_CNN(train_loss, val_loss, f'{name}_CNN_20x20_10000', 'ReLu_and_SGD', batch_size , N_epochs)

    return net


if __name__ == '__main__':
    N_samples = 70_000
    N_epochs = 2000
    noise_pct = 10

    net = main('interpolated', N_samples, N_epochs, noise_pct)

    PATH = 'trained_models/CNN_interpolated_10000.pt'
    torch.save(net, PATH)

    # net = main('2d', N_samples = 10_000)
    # PATH = 'trained_models/CNN_2d_10000.pt'
    # torch.save(net, PATH)






