import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

import produce_and_load_eeg_data
from plot import plot_MSE_NN
from utils import numpy_to_torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(231, 180)
        self.fc2 = nn.Linear(180, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 16)
        # self.fc5 = nn.Linear(16, 4)
        self.fc5 = nn.Linear(16, 3)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, name: str, N_samples : int, normalize_pos_and_radi = True):
        eeg, pos_and_radii = produce_and_load_eeg_data.load_data(N_samples, 'dipole_area')
        mean, std_dev = produce_and_load_eeg_data.load_mean_std(N_samples, 'dipole_area')
        eeg = numpy_to_torch(eeg)

        # Scaling the data
        eeg = (eeg - mean)/std_dev

        # TODO: Ask Torbjørn: When is this not suitable?
        # if normalize_pos_and_radi == True:
        #     pos_and_radii[:,:-1] = (pos_and_radii[:,:-1] - np.mean(pos_and_radii[:,:-1]))/np.std(pos_and_radii[:,:-1])
        #     pos_and_radii[:,-3] = (pos_and_radii[:,-3] - np.mean(pos_and_radii[:,-3]))/np.std(pos_and_radii[:,-3])

        pos_and_radii = numpy_to_torch(pos_and_radii)
        self.eeg, self.pos_and_radii = self.split_data(eeg, pos_and_radii, train_test)

    def split_data(self, eeg, pos_and_radii, train_test):
        eeg_train, eeg_test, pos_and_radii_train, pos_and_radii_test = train_test_split(
            eeg, pos_and_radii, test_size=0.2, random_state=0
        )
        if train_test == 'train':
            return eeg_train, pos_and_radii_train
        elif train_test == 'test':
            return eeg_test, pos_and_radii_test
        else:
            raise ValueError(f'Unknown train_test value {train_test}')

    def __getitem__(self, idx):
        eeg = self.eeg[idx]
        pos_and_radii = self.pos_and_radii[idx]
        return eeg, pos_and_radii

    def __len__(self):
        return self.eeg.shape[0]


def train_epoch(data_loader_train, noise_pct, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, pos_and_radii) in enumerate(data_loader_train):
        pos_and_radii = pos_and_radii[:,:-1]
        optimizer.zero_grad()
        noise = np.random.normal(0, np.std(signal.numpy()) * noise_pct/100, signal.shape)
        signal = numpy_to_torch(signal + noise)
        pred = net(signal)
        loss = criterion(pred, pos_and_radii)
        loss.backward()
        optimizer.step()
        losses[idx] = loss.item()

    mean_loss = np.mean(losses)

    return mean_loss


def test_epoch(data_loader_test, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_test))
    with torch.no_grad():
        for idx, (signal, pos_and_radii) in enumerate(data_loader_test):
            pos_and_radii = pos_and_radii[:,:-1]
            noise = np.random.normal(0, np.std(signal.numpy()) * 10/100, signal.shape)
            signal = numpy_to_torch(signal + noise)
            pred = net(signal)
            loss = criterion(pred, pos_and_radii)
            print('pred')
            print(pred)
            print('pos and radii')
            print(pos_and_radii)
            losses = loss.item()

        mean_loss = np.mean(losses)

    return mean_loss

def main(name: str, N_samples = 10_000, N_epochs = 500, noise_pct = 10):
    print(f'You are now training the network (pos_and_radii) with {N_samples} samples,')
    print(f'and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 30

    net = Net()
    dataset_train = EEGDataset('train', name, N_samples)
    data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    )
    dataset_test = EEGDataset('test', name, N_samples)
    data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    )

    criterion = nn.MSELoss(reduce=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6)

    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    with open(f'results/07.feb/result_pos_and_radii.txt', 'w') as f:
        f.write(f'Samples: {N_samples}, Batch size: {batch_size}, Epochs: {N_epochs}, Noise: {noise_pct} %\n')
        f.write(f'\nEeg Data: data/single_dipole_eeg_{N_samples}.npy')
        f.write(f'\nLocation Data: data/single_dipole_locations_{N_samples}.npy')
        f.write(f'\nMean and std Data: data/single_dipole_eeg_mean_std_{N_samples}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(data_loader_train, noise_pct, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(data_loader_test, optimizer, net, criterion)
        line = f'epoch {epoch:6d}, train loss: {train_loss[epoch]:9.3f}'
        line += f', test loss: {test_loss[epoch]:9.3f}'
        print(line)

        with open(f'results/07.feb/result_pos_and_radii.txt', 'a') as f:
            f.write(f'epoch {epoch:2d}, train loss: {train_loss[epoch]:9.3f}')
            f.write(f', test loss: {test_loss[epoch]:9.3f} \n')

    plot_MSE_NN(train_loss, test_loss, f'v2_NN_pos_and_radii_noise{noise_pct}', 'TanH', batch_size, N_epochs)

    return net

if __name__ == '__main__':
    N_samples = 10_000
    N_epochs = 2000
    noise_pct = 10

    net = main('eeg', N_samples, N_epochs, noise_pct)
    PATH = f'trained_models/v2_NN_pos_and_radii_noise{noise_pct}_10000.pt'
    torch.save(net, PATH)