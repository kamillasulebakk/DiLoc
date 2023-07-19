import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from plot import plot_MSE_NN, plot_MSE_targets
from ffnn import FFNN
from eeg_dataset import EEGDataset


def train_epoch(data_loader_train, optimizer, net, criterion):
    losses = np.zeros(len(data_loader_train))
    for idx, (signal, target_train) in enumerate(data_loader_train):
        optimizer.zero_grad()
        pred = net(signal)
        loss = criterion(pred, target_train) #, N_dipoles)
        # l1_lambda = 0.000001
        #
        # #TODO: fix this list -> tensor hack
        # l1_norm = torch.sum(torch.tensor([torch.linalg.norm(p, 1) for p in net.parameters()]))
        #
        # loss = loss + l1_lambda * l1_norm
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
            loss = criterion(pred, target_test) #, N_dipoles)
            losses[idx] = loss.item()
        mean_loss = np.mean(losses)

        # Adjust the learning rate based on validation loss
        scheduler.step(losses[idx])

    return mean_loss


def main(
    N_samples: int = 10_000,
    N_dipoles: int = 1,
    determine_area: bool = True,
    determine_amplitude=False,
    N_epochs: int = 2000,
    noise_pct: int = 5,
    log_dir: str = 'results',
):
    msg = f'Training network with {N_samples} samples'
    if determine_area:
        msg += ' determining radii'
    else:
        msg += ' without determining radii'
    print(msg)
    print(f'{N_dipoles} dipole(s) and {noise_pct} % noise for {N_epochs} epochs.\n')

    batch_size = 32
    # batch_size = 64
    # batch_size = 128


    net = FFNN(
        [512, 256, 128, 64, 32],
        N_dipoles,
        determine_area,
        determine_amplitude
    )

    dataset_train = EEGDataset(
        'train', determine_area, determine_amplitude, N_samples, N_dipoles
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    dataset_test = EEGDataset(
        'validation', determine_area, determine_amplitude, N_samples, N_dipoles
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
    )

    # criterion = custom_loss
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()


    # lr = 1.5 # Works best for 1 dipole, with amplitude (no radi)
    # lr = 0.001 # Works best for population of dipoles, with amplitude and radii

    # PREDICTS THE BEST
    # lr = 1.5
    # momentum = 0.35
    # weight_decay = 0.1

    # lr = 0.9
    # momentum = 1e-4
    # weight_decay = 1e-5

    # lr = 0.001
    # momentum = 0.35
    # weight_decay = 0.1

    # weight_decay > 0 --> l2/ridge penalty

    # lr = 0.001
    # momentum = 0.35

    lr = 0.001
    momentum = 0.35
    # weight_decay = 0.1
    weight_decay = 0

    # save_file_name: str = f'simple_dipole_l2_less_complicated_network_radius_tanh_{N_samples}_19july_mseloss_MSE_dipole_w_amplitude_{N_epochs}_SGD_lr{lr}_mom{momentum}_wd_{weight_decay}_bs{batch_size}'
    save_file_name: str = f'test123'

    # save_file_name: str = f'adam'
    # lr = 0.001


    optimizer = torch.optim.SGD(net.parameters(), lr, momentum) #, weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr)

    # This one works for radii + amplitude, and amplitude
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10,
    #                           verbose=True, threshold=0.00000001, threshold_mode='rel',
    #                           cooldown=0, min_lr=0, eps=1e-08)

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
        f.write(f'Learning rate: {lr}') #, Momentum: {momentum}, Weight decay: {weight_decay} %\n \n')

        f.write(f'\nEeg Data: data/dipole_area_eeg_{N_samples}_{N_dipoles}.npy')
        f.write(f'\nTarget Data: data/dipole_area_locations_{N_samples}_{N_dipoles}.npy')
        f.write('\n--------------------------------------------------------------\n')

    # Train the model
    status_line = 'Epoch {:4d}/{:4d} | Train: {:6.10f} | Test: {:6.10f} \n'
    for epoch in range(N_epochs):
        train_loss[epoch] = train_epoch(
            data_loader_train, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(
            data_loader_val, net, criterion, scheduler)

        line = status_line.format(
            epoch, N_epochs - 1, train_loss[epoch], test_loss[epoch]
        )
        print(line)
        with open(log_file_name, 'a') as f:
            f.write(line)

        for i, (signal, target) in enumerate(data_loader_val):
            pred = net(signal)
            target_ = target.detach().numpy()
            pred_ = pred.detach().numpy()

            MSE_x[epoch] = np.mean((target_[:][0] - pred_[:][0]) ** 2)
            MSE_y[epoch] = np.mean((target_[:][1] - pred_[:][1]) ** 2)
            MSE_z[epoch] = np.mean((target_[:][2] - pred_[:][2]) ** 2)
            MSE_A[epoch] = np.mean((target_[:][3] - pred_[:][3]) ** 2)

        # print target and predicted values
        if epoch % 100 == 0:
            for i, (signal, target) in enumerate(data_loader_val):
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
        'tanh',
        batch_size,
        N_epochs,
        N_dipoles
    )

    plot_MSE_targets(
        MSE_x,
        MSE_y,
        MSE_z,
        MSE_A,
        'tanh',
        batch_size,
        save_file_name,
        N_dipoles
    )

    # plot_MSE_single_target(
    #     MSE_A,
    #     'tanh',
    #     batch_size,
    #     save_file_name,
    #     N_dipoles
    # )

    torch.save(net, f'trained_models/july/{save_file_name}.pt')


if __name__ == '__main__':
    main(
        N_samples=70000,
        N_dipoles=1,
        determine_area=True,
        determine_amplitude=True,
        N_epochs=20,
        noise_pct=10,
        log_dir='results'
    )
