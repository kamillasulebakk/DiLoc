import json

import numpy as np
import matplotlib.pyplot as plt
from ft.formats import Table

from utils import MSE, MAE
from plot import set_ax_info

class TestResults:
    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self._predictions = predictions.astype(np.float64, copy=True)
        self._targets = targets.astype(np.float64, copy=True)

    def _MAE(self, dimension: int):
        return MAE(self._predictions[:, dimension], self._targets[:, dimension])

    def _MSE(self, dimension: int):
        return MSE(self._predictions[:, dimension], self._targets[:, dimension])


    @property
    def MAE_x(self):
        return self._MAE(0)

    @property
    def MAE_y(self):
        return self._MAE(1)

    @property
    def MAE_z(self):
        return self._MAE(2)

    @property
    def MAE_amplitude(self):
        return self._MAE(3)

    @property
    def MAE_radius(self):
        return self._MAE(4)

    @property
    def MAE_position(self):
        # TODO: Verify that this logic is correct
        return (self.MAE_x + self.MAE_y + self.MAE_z)/3

    @property
    def MSE_x(self):
        return self._MSE(0)

    @property
    def MSE_y(self):
        return self._MSE(1)

    @property
    def MSE_z(self):
        return self._MSE(2)

    @property
    def MSE_amplitude(self):
        return self._MSE(3)

    @property
    def MSE_radius(self):
        return self._MSE(4)

    @property
    def MSE_position(self):
        # TODO: Verify that this logic is correct
        return (self.MSE_x + self.MSE_y + self.MSE_z)/3


    # Two dipoles
    @property
    def MAE_x2(self):
        return self._MAE(3)

    @property
    def MAE_y2(self):
        return self._MAE(4)

    @property
    def MAE_z2(self):
        return self._MAE(5)

    @property
    def MAE_amplitude2(self):
        return self._MAE(7)

    @property
    def MAE_position2(self):
        # TODO: Verify that this logic is correct
        return (self.MAE_x2 + self.MAE_y2 + self.MAE_z2)/3

    @property
    def MSE_x2(self):
        return self._MSE(3)

    @property
    def MSE_y2(self):
        return self._MSE(4)

    @property
    def MSE_z2(self):
        return self._MSE(5)

    @property
    def MSE_amplitude2(self):
        return self._MSE(7)

    @property
    def MSE_position2(self):
        # TODO: Verify that this logic is correct
        return (self.MSE_x2 + self.MSE_y2 + self.MSE_z2)/3



def generate_test_results(predictions: np.ndarray, targets: np.ndarray):
    results = TestResults(predictions, targets)
    d = {
        'MAE_x': results.MAE_x,
        'MAE_y': results.MAE_y,
        'MAE_z': results.MAE_z,
        'MAE_amplitude': results.MAE_amplitude,
        # 'MAE_radius': results.MAE_radius,
        'MAE_position': results.MAE_position,

        'MAE_x2': results.MAE_x2,
        'MAE_y2': results.MAE_y2,
        'MAE_z2': results.MAE_z2,
        # 'MAE_amplitude2': results.MAE_amplitude2,
        # # # 'MAE_radius': results.MAE_radius,
        'MAE_position2': results.MAE_position2,

        'MSE_x': results.MSE_x,
        'MSE_y': results.MSE_y,
        'MSE_z': results.MSE_z,
        'MSE_amplitude': results.MSE_amplitude,
        # 'MSE_radius': results.MSE_radius,
        'MSE_position': results.MSE_position,

        'MSE_x2': results.MSE_x2,
        'MSE_y2': results.MSE_y2,
        'MSE_z2': results.MSE_z,
        # 'MSE_amplitude2': results.MSE_amplitude2,
        # # # 'MSE_radius': results.MSE_radius,
        'MSE_position2': results.MSE_position2
    }
    return d


def print_test_results(d):
    l = [
        ['', 'MAE (mm)', 'MSE (mm^2)', 'RMSE (mm)'],
        ['x', f'{d["MAE_x"]:.3f}', f'{d["MSE_x"]:.3f}', f'{np.sqrt(d["MSE_x"]):.3f}'],
        ['y', f'{d["MAE_y"]:.3f}', f'{d["MSE_y"]:.3f}', f'{np.sqrt(d["MSE_y"]):.3f}'],
        ['z', f'{d["MAE_z"]:.3f}', f'{d["MSE_z"]:.3f}', f'{np.sqrt(d["MSE_z"]):.3f}'],
        ['Position', f'{d["MAE_position"]:.3f}', f'{d["MSE_position"]:.3f}', f'{np.sqrt(d["MSE_position"]):.3f}'],
        # ['Amplitude', f'{d["MAE_amplitude"]:.3f}', f'{d["MSE_amplitude"]:.3f}', f'{np.sqrt(d["MSE_amplitude"]):.3f}'],
        # ['Radius', f'{d["MAE_radius"]:.3f}', f'{d["MSE_radius"]:.3f}', f'{np.sqrt(d["MSE_radius"]):.3f}'],
        ['x2', f'{d["MAE_x2"]:.3f}', f'{d["MSE_x2"]:.3f}', f'{np.sqrt(d["MSE_x2"]):.3f}'],
        ['y2', f'{d["MAE_y2"]:.3f}', f'{d["MSE_y2"]:.3f}', f'{np.sqrt(d["MSE_y2"]):.3f}'],
        ['z2', f'{d["MAE_z2"]:.3f}', f'{d["MSE_z2"]:.3f}', f'{np.sqrt(d["MSE_z2"]):.3f}'],
        ['Position2', f'{d["MAE_position2"]:.3f}', f'{d["MSE_position2"]:.3f}', f'{np.sqrt(d["MSE_position2"]):.3f}'],
        # ['Amplitude2', f'{d["MAE_amplitude2"]:.3f}', f'{d["MSE_amplitude2"]:.3f}', f'{np.sqrt(d["MSE_amplitude2"]):.3f}'],
    ]
    table = Table(l)
    table.write()

def make_histogram(norms, thresholds, x_label, name):
    colors = ['r', 'g', 'b', 'm']

    fig, ax = plt.subplots(figsize=(8, 5))
    hist, bins, _ = ax.hist(norms, bins=10, color='#607c8e', alpha=0.25)

    for i, threshold in enumerate(thresholds):
        ax.axvline(x=threshold, color=colors[i], linestyle='--', label=f'Threshold = {threshold} mm')

    for i, count in enumerate(hist):
        ax.text(bins[i], count + 5, str(int(count)), ha='left', va='bottom', fontsize=10)

    set_ax_info(
        ax,
        xlabel=f'{x_label}',
        ylabel='Frequency',
        title=f'Distribution of Prediction Errors'
    )

    fig.tight_layout()
    fig.savefig(f'plots/histogram_{name}.pdf')
    plt.close(fig)


def make_histogram(num_bins, norms, thresholds, x_label, name):
    color = 'steelblue'  # Choose a color for the histogram bars
    colors = ['r', 'g', 'b', 'm']

    fig, ax = plt.subplots(figsize=(8, 5))
    hist, bins, _ = ax.hist(norms, bins=np.linspace(0, num_bins, num_bins + 1), color=color, alpha=0.7)  # Adjust the bins range

    for i, count in enumerate(hist):
        ax.text(bins[i], count + 5, str(int(count)), ha='left', va='bottom', fontsize=10)

    ax.set_xlim(0, num_bins)  # Set the x-axis limits
    ax.set_xticks(np.arange(0, num_bins + 1, 2))  # Set the x-axis ticks at 2, 4, 6, 8, ...
    set_ax_info(
        ax,
        xlabel=f'{x_label}',
        ylabel='Frequency',
        title=f'Distribution of Prediction Errors'
    )

    fig.tight_layout()
    fig.savefig(f'plots/histogram_{name}.pdf')
    plt.close(fig)


def test_criterea(predictions, targets, name):
    norms = np.linalg.norm(predictions[:,:3] - targets[:,:3], axis=1)


    print(f'Mean Euclidean Distance (MED) is {np.mean(norms)}')

    thresholds = [3, 5, 10, 15]
    results = np.zeros(4)
    for i, threshold in enumerate(thresholds):
        results[i] = sum(norms < threshold)

    results = results/np.shape(predictions)[0]*100

    l = [
        ['', 'MED < 3 mm', 'MED < 5 mm', 'MED < 10 mm', 'MED < 15 mm'],
        ['pos', f'{results[0]:.3f}',
                f'{results[1]:.3f}',
                f'{results[2]:.3f}',
                f'{results[3]:.3f}']
    ]
    table = Table(l)
    table.write()

    make_histogram(28, norms, thresholds, 'Euclidean Distance (mm)', f'position_{name}')


    if name == 'amplitude' or 'area':
        amplitude_absolute_error = np.abs(predictions[:,3] - targets[:,3])
        print(f'Mean Absolute Error (MAE) for amplitude is {np.mean(amplitude_absolute_error)}')
        # within_threshold = sum((norms < 15) & (amplitude_absolute_error < 1.0))

        thresholds = [1, 2, 3]
        results = np.zeros(3)
        for i, threshold in enumerate(thresholds):
            results[i] = sum(amplitude_absolute_error < threshold)

        results = results/np.shape(predictions)[0]*100

        l = [
            ['', 'MAE < 1.0 mA$mu$m', 'MAE < 2.0 mA$mu$m', 'MAE < 3.0 mA$mu$m'],
            ['amplitude', f'{results[0]:.3f}',
                    f'{results[1]:.3f}',
                    f'{results[2]:.3f}']
        ]
        table = Table(l)
        table.write()

        # make_histogram(10, amplitude_absolute_error, thresholds, r'Amplitude Absolute Error ($mA\mu m$)', f'amplitude_{name}')

        results_combined = np.zeros(3)
        for i, threshold in enumerate(thresholds):
            results_combined[i] = sum((norms < 10) & (amplitude_absolute_error < threshold))

        results_combined = results_combined/np.shape(predictions)[0]*100

        l = [
            ['', 'MED < 10 mm & MAE < 1.0 mA$mu$m', 'MED < 10 mm & MAE < 2.0 mA$mu$m', 'MED < 10 mm & MAE < 3.0 mA$mu$m'],
            ['pos & amplitude', f'{results_combined[0]:.3f}',
                    f'{results_combined[1]:.3f}',
                    f'{results_combined[2]:.3f}']
        ]
        table = Table(l)
        table.write()

        # make_histogram(results_combined, amplitude_absolute_error, thresholds, 'Amplitude Absolute Error (mm)', name)

    if name == 'area':
        area_absolute_error = np.abs(predictions[:,4] - targets[:,4])
        print(f'Mean Absolute Error (MAE) for area is {np.mean(area_absolute_error)}')

        # amplitude_absolute_error = np.abs(predictions[:,3] - targets[:,3])

        thresholds = [1, 3, 5]
        results = np.zeros(3)
        for i, threshold in enumerate(thresholds):
            results[i] = sum(area_absolute_error < threshold)

        l = [
            ['', 'MAE < 1.0 mm', 'MAE < 3.0 mm', 'MAE < 5.0 mm'],
            ['area', f'{results[0]/np.shape(predictions)[0]*100:.3f}',
                    f'{results[1]/np.shape(predictions)[0]*100:.3f}',
                    f'{results[2]/np.shape(predictions)[0]*100:.3f}']
        ]
        table = Table(l)
        table.write()

        make_histogram(10, area_absolute_error, thresholds, 'Area Absolute Error (mm)', f'area_{name}')

        results_combined = np.zeros(3)
        for i, threshold in enumerate(thresholds):
            results_combined[i] = sum((norms < 10) & (amplitude_absolute_error < 3) & (area_absolute_error < threshold))

        l = [
            ['', 'MED < 10 mm & MAE < 3.0 mA$mu$m & MAE < 1.0 mm', 'MED < 10 mm & MAE < 3.0 mA$mu$m & MAE < 3.0 mm', 'MED < 10 mm & MAE < 3.0 mA$mu$m & MAE < 5.0 mm'],
            ['pos & amplitude & radius', f'{results_combined[0]/np.shape(predictions)[0]*100:.3f}',
                    f'{results_combined[1]/np.shape(predictions)[0]*100:.3f}',
                    f'{results_combined[2]/np.shape(predictions)[0]*100:.3f}']
        ]
        table = Table(l)
        table.write()


    # if name == 'two_dipoles':



def save_test_results(d, filename: str):
    with open(filename, 'w', encoding='UTF-8') as f:
        json.dump(d, f, indent=2)


def load_test_results(filename: str):
    with open(filename, 'r', encoding='UTF-8') as f:
        return json.load(f)
