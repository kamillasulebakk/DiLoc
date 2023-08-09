import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt

from plot import set_ax_info

def sort_arrays(predictions, target):
    # Get the indices that would sort the target array based on the fourth element in each row
    sorted_indices = np.argsort(target[:, 3])

    # Use the sorted indices to sort both predictions and target arrays
    sorted_predictions = predictions[sorted_indices, :]
    sorted_target = target[sorted_indices, :]

    return sorted_predictions, sorted_target


def plot_error_amplitude(predictions, targets):
    sorted_predictions, sorted_targets = sort_arrays(predictions, targets)

    np.mean((predictions - targets)**2, axis=1)
    mae_values = np.mean(np.abs(sorted_predictions[:, :3] - sorted_targets[:, :3]), axis=1)

    mse_values = np.mean((sorted_predictions[:, :3] - sorted_targets[:, :3])**2, axis=1)

    amplitude = targets[:, 3]

    fig, ax = plt.subplots()
     # Plot the MSE against the amplitude
    ax.scatter(amplitude, mse_values)
    set_ax_info(
        ax,
        xlabel='Amplitude',
        ylabel='MSE',
        title=f'MSE between Predictions and Target as function of Amplitude'
    )
    fig.tight_layout()
    fig.savefig(f'plots/mse_amplitude.pdf')
    plt.close(fig)

    fig, ax = plt.subplots()
     # Plot the MAE against the amplitude
    ax.scatter(amplitude, mae_values)
    set_ax_info(
        ax,
        xlabel='Amplitude',
        ylabel='MAE',
        title=f'MAE between Predictions and Target as function of Amplitude'
    )
    fig.tight_layout()
    fig.savefig(f'plots/mae_amplitude.pdf')
    plt.close(fig)

