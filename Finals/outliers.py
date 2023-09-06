import numpy as np
import matplotlib.pyplot as plt
from plot import set_ax_info
import seaborn as sns

data = np.load('data/simple_70000_1_eeg_train-validation.npy')
data_min = np.min(data, axis=1)
data_max = np.max(data, axis=1)


data_sets = [data_min, data_max]
name = ['Min', 'Max']
pos_legend = ['lower right', 'upper right']

# Load the 'tab10' color palette from Seaborn
sns.set_palette('tab10')

for i, data in enumerate(data_sets):
    mean = np.mean(data)
    std = np.std(data)

    threshold = 3 * std

    fig, ax = plt.subplots()

    # Create a scatter plot using the default palette
    ax.scatter(range(len(data)), data)

    # Plot the mean and threshold lines
    ax.axhline(mean, color='r', linestyle='dashed', linewidth=2, label='Mean')
    ax.axhline(mean + threshold * std, color='g', linestyle='dotted', linewidth=2, label=r'Upper Threshold, 3$\sigma$')
    ax.axhline(mean - threshold * std, color='g', linestyle='dotted', linewidth=2, label=r'Lower Threshold, 3$\sigma$')

    # Add labels and legend
    ax.legend()

    set_ax_info(
        ax,
        xlabel='Number of Samples',
        ylabel=f'Measured Signal' + r' [$\mu$V]',
        title=f'Data Visualization, {name[i]} EEG Measures',
        legend=False
    )
    ax.legend(loc=pos_legend[i])

    fig.tight_layout()
    fig.savefig(f'plots/data_visualization_{name[i]}.pdf')
    plt.close(fig)