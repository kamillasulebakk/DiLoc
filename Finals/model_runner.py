import os
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from plot import plot_MSE_NN, plot_MSE_targets
from ffnn import FFNN, number_of_output_values
from eeg_dataset import EEGDataset, generate_log_filename


def train_epoch(data_loader, optimizer, net, criterion):
    total_loss = 0.0
    for eeg, target in data_loader:
        optimizer.zero_grad()
        pred = net(eeg)
        loss = criterion(pred, target, net, is_training=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss/len(data_loader)
    return mean_loss


def val_epoch(data_loader, net, criterion, scheduler):
    total_loss = 0.0    # NB! Will turn into torch.Tensor
    SE_targets = 0.0    # NB! Will turn into torch.Tensor
    total_number_of_samples = 0
    with torch.no_grad():
        for eeg, target in data_loader:
            pred = net(eeg)
            loss = criterion(pred, target, net, is_training=False)
            total_loss += loss

            SE_targets += ((target - pred)**2).sum(dim=0)
            total_number_of_samples += target.shape[0]

        # Adjust the learning rate based on validation loss
        scheduler.step(total_loss)

    mean_loss = total_loss.item()/len(data_loader)
    MSE_targets = SE_targets.numpy()/total_number_of_samples
    return mean_loss, MSE_targets


class Logger:
    def __init__(self, parameters):
        self._parameters = parameters
        self._log_fname = os.path.join(
            'results', parameters['log_fname'] + '.txt'
        )
        self._write_line(self._header_line())
        self._start_time = time.perf_counter()

    def _header_line(self):
        lines = [f'{key}: {value}' for key, value in self._parameters.items()]
        lines.append('-'*60 + '\n')
        return '\n'.join(lines)

    def _write_line(self, line):
        print(line, end='')
        with open(self._log_fname, 'a', encoding='UTF-8') as f:
            f.write(line)

    def status(self, epoch: int, loss: float, val: float):
        status_line = 'Epoch {:4d}/{:4d} | Train: {:13.8f} | Validation: {:13.8f} | Time: {:10.3} s\n'
        line = status_line.format(
            epoch,
            self._parameters['N_epochs'] - 1,
            loss,
            val,
            (time.perf_counter() - self._start_time)*1000   # convert to seconds
        )
        self._write_line(line)

    def print_predictions(self, predictions, targets):
        lines = []
        for pred, target in zip(predictions, targets):
            lines.append(f'Predicted: {pred.detach().numpy()}')
            lines.append(f'Target:    {target.numpy()}\n')
        self._write_line('\n'.join(lines) + '\n')


class Loss:
    def __init__(self, l1_lambda):
        self._lambda = l1_lambda
        self._mse = torch.nn.MSELoss()

    def __call__(self, predicted, target, net, is_training: bool):
        result = self._mse(predicted, target)
        if is_training:
            # TODO: fix this list -> tensor hack
            l1_norm = torch.sum(
                torch.tensor([torch.linalg.norm(p, 1) for p in net.parameters()])
            )
            result += self._lambda * l1_norm
        return result


def run_model(parameters):
    parameters['log_fname'] = generate_log_filename(parameters)
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

    criterion = Loss(parameters['l1_lambda'])

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
    val_loss = np.zeros_like(train_loss)
    MSE_targets = np.zeros((
        parameters['N_epochs'],
        number_of_output_values(
            parameters['determine_amplitude'],
            parameters['determine_area']
        )
    ))

    # Train the model
    for epoch in range(parameters['N_epochs']):
        train_loss[epoch] = train_epoch(
            data_loader_train, optimizer, net, criterion)
        val_loss[epoch], MSE_targets[epoch] = val_epoch(
            data_loader_val, net, criterion, scheduler)
        logger.status(epoch, train_loss[epoch], val_loss[epoch])

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

    plot_MSE_NN(
        train_loss,
        val_loss,
        parameters['log_fname'],
        'tanh',
        parameters['batch_size'],
        parameters['N_epochs'],
        parameters['N_dipoles']
    )

    plot_MSE_targets(
        MSE_targets,
        parameters['batch_size'],
        parameters['log_fname'],
        parameters['N_dipoles']
    )

    torch.save(net, f'trained_models/{parameters["log_fname"]}.pt')
