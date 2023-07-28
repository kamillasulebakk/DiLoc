import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_data import load_data_files
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

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)     # flatten all dimensions except batch
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, name: str, N_samples: int, noise_pct: int = 10):

        # Use eeg data from selfdefined 2d matrix
        if name == '2d':
            eeg, pos_list = load_data_files(N_samples, 'simple_dipole', '2d', 1)


        # Use eeg data from 2d interpolated data
        elif name == 'interpolated':
            eeg, pos_list = load_data_files(N_samples, 'simple_dipole', 'interpolated', 1)

        # Scaling the data
        eeg = (eeg - np.mean(eeg))/np.std(eeg)

        eeg_matrix = utils.numpy_to_torch(eeg)
        pos_list = utils.numpy_to_torch(pos_list)

        self.eeg_matrix, self.pos_list = self.split_data(eeg_matrix, pos_list, train_test, noise_pct)


    def split_data(self, eeg, pos_list, train_test, noise_pct):
        eeg_train, eeg_test, pos_list_train, pos_list_test = train_test_split(
            eeg, pos_list, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            noise = torch.normal(0, torch.std(eeg_train) * noise_pct/100, size=eeg_train.shape)
            eeg_train += noise
            return eeg_train, pos_list_train
        elif train_test == 'test':
            noise = torch.normal(0, torch.std(eeg_test) * noise_pct/100, size=eeg_test.shape)
            eeg_test += noise
            return eeg_test, pos_list_test
        else:
            raise ValueError(f'Unknown train_test value {train_test}')


    def __getitem__(self, idx):
        eeg = self.eeg_matrix[idx]
        pos = self.pos_list[idx]
        return eeg, pos


    def __len__(self):
        return self.eeg_matrix.shape[0]


def train_epoch(data_loader, noise_pct, optimizer, net, criterion):
    total_loss = 0.0
    for eeg, target in data_loader:
        eeg = eeg.unsqueeze(1)
        optimizer.zero_grad()
        pred = net(eeg)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss/len(data_loader)
    return mean_loss


def test_epoch(data_loader, optimizer, net, criterion, scheduler):
    total_loss = 0.0    # NB! Will turn into torch.Tensor
    SE_targets = 0.0    # NB! Will turn into torch.Tensor
    total_number_of_samples = 0
    with torch.no_grad():
        for eeg, target in data_loader:
            eeg = eeg.unsqueeze(1)
            pred = net(eeg)
            loss = criterion(pred, target)
            total_loss += loss

            SE_targets += ((target - pred)**2).sum(dim=0)
            total_number_of_samples += target.shape[0]

        # Adjust the learning rate based on validation loss
        scheduler.step(total_loss)

    mean_loss = total_loss.item()/len(data_loader)
    MSE_targets = SE_targets.numpy()/total_number_of_samples

    return mean_loss


def main(name: str, N_samples = 10_000, N_epochs = 2000, noise_pct = 10):
    print(f'You are now training the CNN with {N_samples} samples,')
    print(f'and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 30

    net = CNN()
    dataset_train = EEGDataset('train', name, N_samples, noise_pct)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataset_test = EEGDataset('test', name, N_samples, noise_pct)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.009)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                patience=50, verbose=True, threshold=0.0001,
                                threshold_mode='rel', cooldown=0, min_lr=0,
                                eps=1e-08)

    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    with open(f'results/restult_single_dipole_CNN.txt', 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'\nEeg Data: data/single_dipole_eeg_{N_samples}.npy')
        f.write(f'\nLocation Data: data/single_dipole_locations_{N_samples}.npy')
        f.write(f'\nMean and std Data: data/single_dipole_eeg_mean_std_{N_samples}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(data_loader_train, noise_pct, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(data_loader_test, optimizer, net, criterion, scheduler)
        line = f'epoch {epoch:6d}, train loss: {train_loss[epoch]:9.3f}'
        line += f', test loss: {test_loss[epoch]:9.3f}'
        print(line)

        with open(f'results/restult_single_dipole_CNN.txt', 'a') as f:
            f.write(f'epoch {epoch:2d}, train loss: {train_loss[epoch]:9.3f}')
            f.write(f', test loss: {test_loss[epoch]:9.3f} \n')

    plot_MSE_CNN(train_loss, test_loss, f'{name}_CNN_20x20_10000', 'ReLu_and_SGD', batch_size , N_epochs)

    return net


if __name__ == '__main__':
    N_samples = 50_000
    N_epochs = 500
    noise_pct = 10

    net = main('interpolated', N_samples, N_epochs, noise_pct)

    PATH = 'trained_models/CNN_interpolated_10000.pt'
    torch.save(net, PATH)

    # net = main('2d', N_samples = 10_000)
    # PATH = 'trained_models/CNN_2d_10000.pt'
    # torch.save(net, PATH)






