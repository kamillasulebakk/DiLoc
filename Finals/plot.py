import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)
# for e.g. \text command
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

big_data_path = '/Users/Kamilla/Documents/DiLoc-data'


def set_ax_info(ax, xlabel, ylabel, title=None, zlabel=None, legend=True):
    """Write title and labels on an axis with the correct fontsizes.

    Args:
        ax (matplotlib.axis): the axis on which to display information
        xlabel (str): the desired label on the x-axis
        ylabel (str): the desired label on the y-axis
        title (str, optional): the desired title on the axis
            default: None
        zlabel (str, optional): the desired label on the z-axis for 3D-plots
            default: None
        legend (bool, optional): whether or not to add labels/legend
            default: True
    """
    if zlabel == None:
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)
        # ax.ticklabel_format(style='plain')
    else:
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_zlabel(zlabel, fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.ticklabel_format(style='scientific', scilimits = (-2, 2))
    if title != None:
        ax.set_title(title, fontsize=20)
    if legend:
        ax.legend(fontsize=15)


def plot_MSE_targets_2_dipoles(
    MSE_x1,
    MSE_y1,
    MSE_z1,
    MSE_A1,
    MSE_x2,
    MSE_y2,
    MSE_z2,
    MSE_A2,
    act_func,
    batch_size,
    NN,
    N_dipoles
):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE)', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(MSE_x1), label='no 1, x position [mm]')
    ax.plot(np.log(MSE_y1), label='no 1, y position [mm]')
    ax.plot(np.log(MSE_z1), label='no 1, z position [mm]')
    ax.plot(np.log(MSE_A1), label='no 1, Amplitude [nA um]')
    ax.plot(np.log(MSE_x2), label='no 2, x position [mm]')
    ax.plot(np.log(MSE_y2), label='no 2, y position [mm]')
    ax.plot(np.log(MSE_z2), label='no 2, z position [mm]')
    ax.plot(np.log(MSE_A2), label='no 2, Amplitude [nA um]')
    ax.legend(fontsize=18)
    plt.tight_layout()
    fig.savefig(f'plots/custom_loss_targets_{NN}_2_dipoles.pdf')


def plot_MSE_targets(targets, batch_size, filename, N_dipoles):
    fig, ax = plt.subplots()
    labels = ['$x$ position', '$y$ position', '$z$ position', r'Amplitude', 'Radius']
    for target, label in zip(targets.T, labels):
        ax.plot(np.log(target), label=label)
    set_ax_info(
        ax,
        xlabel='Number of epochs',
        ylabel='ln(Loss)',
        title=f'Custom loss for normalized validation data'
    )
    fig.tight_layout()
    fig.savefig(f'plots/Custom_Loss_mse_targets_{filename}.pdf')
    plt.close(fig)

def plot_MSE_NN(train_loss, test_loss, filename, act_func, batch_size, num_epochs, N_dipoles):
    fig, ax = plt.subplots()
    ax.plot(np.log(train_loss), label='Train')
    ax.plot(np.log(test_loss), label='Validation')
    set_ax_info(
        ax,
        xlabel='Number of epochs',
        ylabel='ln(Loss)',
        title=f'Custom loss for normalized train and validation data',
    )
    fig.tight_layout()
    fig.savefig(f'plots/Custom_Loss_{filename}.pdf')
    plt.close(fig)

def plot_R2_NN(train_R2, test_R2, NN, act_func, batch_size, num_epochs, name = "NN"):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'R2 score for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    epoch_array = np.linspace(15, num_epochs, num_epochs-15)
    ax.plot(epoch_array, train_R2[15:], label='Train')
    ax.plot(epoch_array, test_R2[15:], label='Test')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/R2_{NN}_{act_func}_{batch_size}_{num_epochs}.png')


def plot_MSE_CNN(train_loss, test_loss, NN, act_func, batch_size, num_epochs):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f'MSE for train and test data using {act_func} with {batch_size} batches', fontsize=20)
    ax.set_xlabel('Number of epochs', fontsize=18)
    ax.set_ylabel('ln(MSE) [mm]', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(np.log(train_loss), label='Train')
    ax.plot(np.log(test_loss), label='Test')
    ax.legend(fontsize=18)
    fig.savefig(f'plots/07.feb/MSE_CNN_{NN}_{act_func}_{batch_size}_{num_epochs}.png')




