import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from plot import plot_MSE_NN, plot_MSE_targets
from ffnn import FFNN
from eeg_dataset import EEGDataset


def train_epoch(data_loader, optimizer, net, criterion):
    total_loss = 0.0
    for eeg, target in data_loader:
        optimizer.zero_grad()
        pred = net(eeg)
        loss = criterion(pred, target)
        # l1_lambda = 0.000001
        #
        # #TODO: fix this list -> tensor hack
        # l1_norm = torch.sum(torch.tensor([torch.linalg.norm(p, 1) for p in net.parameters()]))
        #
        # loss = loss + l1_lambda * l1_norm
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss/len(data_loader)
    return mean_loss


def test_epoch(data_loader, net, criterion, scheduler):
    total_loss = 0.0
    with torch.no_grad():
        for eeg, target in data_loader:
            pred = net(eeg)
            loss = criterion(pred, target)
            total_loss += loss
        # Adjust the learning rate based on validation loss
        scheduler.step(loss)
    mean_loss = total_loss.item()/len(data_loader)
    return mean_loss


batch_sizes = [32, 64, 128]


class Logger:
    def __init__(self, parameters):
        self._parameters = parameters
        fname = os.path.join(
            'results', self._parameters['log_fname'] + '.txt'
        )
        i = 1
        while os.path.isfile(fname):
            fname = os.path.join(
                'results', self._parameters['log_fname'] + f'_({i}).txt'
            )
            i += 1
        self._log_fname = fname
        self._write_line(self._header_line())

    def _header_line(self):
        lines = [f'{key}: {value}' for key, value in self._parameters.items()]
        lines.append('-'*60 + '\n')
        return '\n'.join(lines)

    def _write_line(self, line):
        print(line, end='')
        with open(self._log_fname, 'a', encoding='UTF-8') as f:
            f.write(line)

    def status(self, epoch: int, loss: float, val: float):
        status_line = 'Epoch {:4d}/{:4d} | Train: {:13.8f} | Test: {:13.8f}\n'
        line = status_line.format(
            epoch,
            self._parameters['N_epochs'] - 1,
            loss,
            val
        )
        self._write_line(line)

    def print_predictions(self, predictions, targets):
        lines = []
        for pred, target in zip(predictions, targets):
            lines.append(f'Predicted: {pred.detach().numpy()}')
            lines.append(f'Target:    {target.numpy()}\n')
        self._write_line('\n'.join(lines) + '\n')


def run_model(parameters):
    logger = Logger(parameters)
    net = FFNN(parameters)
    data_loader_train = torch.utils.data.DataLoader(
        EEGDataset('train', parameters),
        batch_size=parameters['batch_size'],
        shuffle=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        EEGDataset('validation', parameters),
        batch_size=parameters['batch_size'],
        shuffle=False,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.SGD(
        net.parameters(),
        parameters['learning_rate'],
        parameters['momentum'],
        parameters['weight_decay']
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                patience=50, verbose=True, threshold=0.0001,
                                threshold_mode='rel', cooldown=0, min_lr=0,
                                eps=1e-08)

    train_loss = np.zeros(parameters['N_epochs'])
    test_loss = np.zeros_like(train_loss)

    MSE_targets = np.zeros((4, parameters['N_epochs']))
    MSE_x, MSE_y, MSE_z, MSE_A = MSE_targets

    # Train the model
    for epoch in range(parameters['N_epochs']):
        train_loss[epoch] = train_epoch(
            data_loader_train, optimizer, net, criterion)
        test_loss[epoch] = test_epoch(
            data_loader_val, net, criterion, scheduler)
        logger.status(epoch, train_loss[epoch], test_loss[epoch])

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
            preds = []
            targets = []
            for i, (eeg, target) in enumerate(data_loader_val):
                pred = net(eeg)
                preds.append(pred[0])
                targets.append(target[0])
                if i == 2:
                    break
            logger.print_predictions(preds, targets)

    # plot_MSE_NN(
        # train_loss,
        # test_loss,
        # save_file_name,
        # 'tanh',
        # batch_size,
        # N_epochs,
        # N_dipoles
    # )

    # plot_MSE_targets(
        # MSE_x,
        # MSE_y,
        # MSE_z,
        # MSE_A,
        # 'tanh',
        # batch_size,
        # save_file_name,
        # N_dipoles
    # )

    # plot_MSE_single_target(
    #     MSE_A,
    #     'tanh',
    #     batch_size,
    #     save_file_name,
    #     N_dipoles
    # )

    torch.save(net, f'trained_models/july/{parameters["log_fname"]}.pt')


def main():
    parameters = {
        'N_samples': 70_000,
        'N_dipoles': 1,
        'determine_amplitude': False,
        'determine_area': False,
        'hidden_layers': [512, 256, 128, 64, 32],
        'batch_size': 32,
        'learning_rate': 0.001,
        'momentum': 0.35,
        'weight_decay': 0.1,
        'N_epochs': 20,
        'noise_pct': 10,
        'log_fname': 'test12345',
    }
    run_model(parameters)


if __name__ == '__main__':
    main()
