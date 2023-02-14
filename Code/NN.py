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
    def __init__(self, N_dipoles, name):
        super().__init__()
        self.fc1 = nn.Linear(231, 180)
        self.fc2 = nn.Linear(180, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 16)
        if name == 'dipole_area':
            self.fc5 = nn.Linear(16, 4)
        else:
            self.fc5 = nn.Linear(16, 3*N_dipoles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, train_test: str, name: str, N_samples: int, N_dipoles: int):
        eeg, pos_list = produce_and_load_eeg_data.load_data(N_samples, name, num_dipoles = N_dipoles)
        mean, std_dev = produce_and_load_eeg_data.load_mean_std(N_samples, f'{name}_{N_dipoles}')

        eeg = numpy_to_torch(eeg)

        if N_dipoles > 1:
            pos_list = np.reshape(pos_list, (N_samples, 3*N_dipoles))

        if name == 'dipole_area':
            pos_list[:,:-1] = (pos_list[:,:-1] - np.mean(pos_list[:,:-1]))/np.std(pos_list[:,:-1])
            pos_list[:,-3] = (pos_list[:,-3] - np.mean(pos_list[:,-3]))/np.std(pos_list[:,-3])

        pos_list = numpy_to_torch(pos_list)

        eeg = (eeg - mean)/std_dev


        self.eeg, self.pos_list = self.split_data(eeg, pos_list, train_test)

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
        eeg = self.eeg[idx]
        pos = self.pos_list[idx]
        return eeg, pos

    def __len__(self):
        return self.eeg.shape[0]


def train_epoch(data_loader_train, noise_pct, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, position) in enumerate(data_loader_train):
        optimizer.zero_grad()
        noise = np.random.normal(0, np.std(signal.numpy()) * noise_pct/100, signal.shape)
        signal = numpy_to_torch(signal + noise)
        pred = net(signal)
        loss = criterion(pred, position)
        l1_lambda = 0.001
        l1_norm = np.sum(torch.linalg.norm(p, 1) for p in net.parameters())
        loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        losses[idx] = loss.item()
    mean_loss = np.mean(losses)
    return mean_loss


def test_epoch(data_loader_test, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_test))
    with torch.no_grad():
        for idx, (signal, position) in enumerate(data_loader_test):
            pred = net(signal)
            loss = criterion(pred, position)
            losses[idx] = loss.item()
        print(position)
        print(pred)
        mean_loss = np.mean(losses)

    return mean_loss

def main(name: str, N_samples = 10_000, N_dipoles = 1, N_epochs = 2000, noise_pct = 10):
    print(f'You are now training the network {name} with {N_samples} samples,')
    print(f'{N_dipoles} dipole(s) and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 30

    net = Net(N_dipoles, name)
    dataset_train = EEGDataset('train', name, N_samples, N_dipoles)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    dataset_test = EEGDataset('test', name, N_samples, N_dipoles)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=1e-6, weight_decay = 1e-5) #weight_decay > 0 --> l2/ridge penalty
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, momentum=1e-6, weight_decay = 1e-5) #weight_decay > 0 --> l2/ridge penalty


    train_loss = np.zeros(N_epochs)
    test_loss = np.zeros(N_epochs)

    with open(f'results/13.feb/result_{name}_{N_dipoles}_NN_{N_samples}_l1_l2_5mm.txt', 'w') as f:
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

        with open(f'results/13.feb/result_{name}_{N_dipoles}_NN_{N_samples}_l1_l2_5mm.txt', 'a') as f:
            f.write(f'epoch {epoch:2d}, train loss: {train_loss[epoch]:9.3f}')
            f.write(f', test loss: {test_loss[epoch]:9.3f} \n')

    plot_MSE_NN(train_loss, test_loss, f'NN_{name}_{N_dipoles}_noise_{noise_pct}_{N_samples}_l1_l2_5mm', 'TanH', batch_size, N_epochs, name = name)

    return net


if __name__ == '__main__':
    N_samples = 10_000
    N_dipoles = 1
    N_epochs = 500
    noise_pct = 15
    name = 'dipole_area'

    net = main(name, N_samples, N_dipoles, N_epochs, noise_pct)

    PATH = f'trained_models/NN_{name}_{N_dipoles}_noise_{noise_pct}_{N_epochs}_{N_samples}_l1_l2_5mm.pt'
    torch.save(net, PATH)

