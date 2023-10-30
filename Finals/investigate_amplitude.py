import numpy as np
from lfpykit.eegmegcalc import NYHeadModel
import matplotlib.pyplot as plt

from plot import set_ax_info
import seaborn as sns


plt.style.use('seaborn')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.serif": ["Computer Modern"]}
)

palette = sns.color_palette("deep")
sns.set_palette(palette)

def sort_arrays(predictions, target, indx = 3):
    # Get the indices that would sort the target array based on the fourth element in each row
    sorted_indices = np.argsort(target[:, indx])

    # Use the sorted indices to sort both predictions and target arrays
    sorted_predictions = predictions[sorted_indices, :]
    sorted_target = target[sorted_indices, :]

    return sorted_predictions, sorted_target


def plot_error_amplitude(predictions, targets):
    sorted_predictions, sorted_targets = sort_arrays(predictions, targets)

    mae_values = np.mean(np.abs(sorted_predictions[:, :3] - sorted_targets[:, :3]), axis=1)

    mse_values = np.mean((sorted_predictions[:, :3] - sorted_targets[:, :3])**2, axis=1)

    amplitude = targets[:, 3]

    correlation_mae = np.corrcoef(amplitude, mae_values)[0, 1]
    correlation_mse = np.corrcoef(amplitude, mse_values)[0, 1]


    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    # Plot the MSE against the amplitude
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.25)
    ax.scatter(amplitude, mse_values, c=palette[3])
    ax.text(0.5, 0.9, f'Correlation coefficient: {correlation_mse:.2f}', transform=ax.transAxes,
         horizontalalignment='left', verticalalignment='top',
         fontsize=15, color='black', bbox=props)

    set_ax_info(
        ax,
        xlabel=f'Magnitude [mAm]',
        ylabel=r'SE [mm$^2$]',
        title=f'SE Between Predicted and Target Position \nas Function of Magnitude'
    )

    # fig.tight_layout()
    ax.set_yscale('log')
    ax.margins(x=0.1)
    fig.savefig(f'plots/mse_amplitude.pdf')
    plt.close(fig)

    # Plot the MAE against the amplitude
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.25)
    ax.scatter(amplitude, mae_values, c=palette[3])
    ax.text(0.5, 0.9, f'Correlation coefficient: {correlation_mae:.2f}', transform=ax.transAxes,
         horizontalalignment='left', verticalalignment='top',
         fontsize=15, color='black', bbox=props)

    set_ax_info(
        ax,
        xlabel=f'Magnitude [mAm]',
        ylabel='AE [mm]',
        title=f'AE Between Predicted and Target Position \nas Function of Magnitude'
    )
    # fig.tight_layout()
    ax.set_yscale('log')
    ax.margins(x=0.1)
    fig.savefig(f'plots/mae_amplitude.pdf')
    plt.close(fig)

def plot_error_area(predictions, targets):
    '''
    Plot the MED of center as function of area
    '''

    sorted_predictions, sorted_targets = sort_arrays(predictions, targets, 4)

    # med_values = np.mean(np.linalg.norm(sorted_predictions[:, :3] - sorted_targets[:, :3]), axis=1)
    # print(med_values)

    mae_values = np.mean(np.abs(sorted_predictions[:, :3] - sorted_targets[:, :3]), axis=1)

    mse_values = np.mean((sorted_predictions[:, :3] - sorted_targets[:, :3])**2, axis=1)
    radius = targets[:, 4]

    correlation_mae = np.corrcoef(radius, mae_values)[0, 1]
    correlation_mse = np.corrcoef(radius, mse_values)[0, 1]

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.25)
    # Plot the MSE against the radius
    ax.scatter(radius, mse_values, c=palette[3])
    ax.text(0.5, 0.9, f'Correlation coefficient: {correlation_mse:.2f}', transform=ax.transAxes,
         horizontalalignment='left', verticalalignment='top',
         fontsize=15, color='black', bbox=props)

    set_ax_info(
        ax,
        xlabel=r'Radius [mm]',
        ylabel=r'SE [mm$^2$]',
        title=f'SE Between Predicted and Target Center \nas Function of Radius'
    )
    # fig.tight_layout()
    ax.set_yscale('log')
    ax.margins(x=0.1)
    fig.savefig(f'plots/mse_area.pdf')
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(left=0.25)
    # Plot the MAE against the radius
    ax.scatter(radius, mae_values, c=palette[3])
    ax.text(0.5, 0.9, f'Correlation coefficient: {correlation_mae:.2f}', transform=ax.transAxes,
         horizontalalignment='left', verticalalignment='top',
         fontsize=15, color='black', bbox=props)


    set_ax_info(
        ax,
        xlabel=r'Radius [mm]',
        ylabel='AE [mm]',
        title=f'AE Between Predicted and Target Center \nas Function of Radius'
    )
    # fig.tight_layout()
    ax.set_yscale('log')
    ax.margins(x=0.1)
    fig.savefig(f'plots/mae_area.pdf')
    plt.close(fig)



