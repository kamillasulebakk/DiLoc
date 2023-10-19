import json

import numpy as np
import matplotlib.pyplot as plt
from ft.formats import Table

from utils import MSE, MAE
from plot import set_ax_info

import itertools


class TestResults:
    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self._predictions = predictions.astype(np.float64, copy=True)
        self._targets = targets.astype(np.float64, copy=True)
        self.N_outputs = np.shape(predictions)[1]
        self.N_samples = np.shape(predictions)[0]

    def one_sample(self, predicted, target):
        N_dipoles = 2
        N_outputs = len(predicted) // N_dipoles
        combinations = itertools.permutations(range(N_dipoles), N_dipoles) # [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
        result = 1e99

        individual_dist = []

        for pred_indices in combinations:
            tmp = 0
            for i, j in enumerate(pred_indices):
                euc_dist = self.one_dipole(
                    predicted[N_outputs*j:N_outputs*(j + 1)],
                    target[N_outputs*i:N_outputs*(i + 1)]
                )
                individual_dist.append(euc_dist)
                tmp += euc_dist

            result = min(result, tmp)

        sum_1 = individual_dist[0] + individual_dist[1]
        sum_2 = individual_dist[2] + individual_dist[3]

        if sum_1 > sum_2:
            individual_dist.remove(individual_dist[0])
            individual_dist.remove(individual_dist[0])
            removed_idx = 0
        else:
            individual_dist.remove(individual_dist[-1])
            individual_dist.remove(individual_dist[-1])
            removed_idx = 1

        return removed_idx

    def one_dipole(self, predicted, target):
        euc_dist = np.linalg.norm(predicted[:3] - target[:3])
        return euc_dist

    def _MAE(self, dimension: int):
        if self.N_outputs == 6:
            new_order_predictions = np.copy(self._predictions)
            for i in range(self.N_samples):
                removed_idx = self.one_sample(self._predictions[i], self._targets[i])
                if removed_idx == 0:
                    new_order_predictions[i, 0:3] = self._predictions[i, 3:6]
                    new_order_predictions[i, 3:6] = self._predictions[i, 0:3]

            return MAE(new_order_predictions[:, dimension], self._targets[:, dimension])
        else:
            return MAE(self._predictions[:, dimension], self._targets[:, dimension])

    def _MSE(self, dimension: int):
        if self.N_outputs == 6:
            new_order_predictions = np.copy(self._predictions)
            for i in range(self.N_samples):
                removed_idx = self.one_sample(self._predictions[i], self._targets[i])
                if removed_idx == 0:
                    new_order_predictions[i, 0:3] = self._predictions[i, 3:6]
                    new_order_predictions[i, 3:6] = self._predictions[i, 0:3]
            return MSE(new_order_predictions[:, dimension], self._targets[:, dimension])
        else:
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
        # 'MAE_amplitude': results.MAE_amplitude,
        # 'MAE_radius': results.MAE_radius,
        'MAE_position': results.MAE_position,

        # 'MAE_x2': results.MAE_x2,
        # 'MAE_y2': results.MAE_y2,
        # 'MAE_z2': results.MAE_z2,
        # # 'MAE_amplitude2': results.MAE_amplitude2,
        # # 'MAE_radius': results.MAE_radius,
        # 'MAE_position2': results.MAE_position2,

        'MSE_x': results.MSE_x,
        'MSE_y': results.MSE_y,
        'MSE_z': results.MSE_z,
        # 'MSE_amplitude': results.MSE_amplitude,
        # 'MSE_radius': results.MSE_radius,
        'MSE_position': results.MSE_position,

        # 'MSE_x2': results.MSE_x2,
        # 'MSE_y2': results.MSE_y2,
        # 'MSE_z2': results.MSE_z,
        # # 'MSE_amplitude2': results.MSE_amplitude2,
        # # 'MSE_radius': results.MSE_radius,
        # 'MSE_position2': results.MSE_position2
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
        # ['x2', f'{d["MAE_x2"]:.3f}', f'{d["MSE_x2"]:.3f}', f'{np.sqrt(d["MSE_x2"]):.3f}'],
        # ['y2', f'{d["MAE_y2"]:.3f}', f'{d["MSE_y2"]:.3f}', f'{np.sqrt(d["MSE_y2"]):.3f}'],
        # ['z2', f'{d["MAE_z2"]:.3f}', f'{d["MSE_z2"]:.3f}', f'{np.sqrt(d["MSE_z2"]):.3f}'],
        # ['Position2', f'{d["MAE_position2"]:.3f}', f'{d["MSE_position2"]:.3f}', f'{np.sqrt(d["MSE_position2"]):.3f}'],
        # ['Amplitude2', f'{d["MAE_amplitude2"]:.3f}', f'{d["MSE_amplitude2"]:.3f}', f'{np.sqrt(d["MSE_amplitude2"]):.3f}'],
    ]
    table = Table(l)
    table.write()

# def make_histogram(norms, thresholds, x_label, name):
#     colors = ['r', 'g', 'b', 'm']
#
#     fig, ax = plt.subplots(figsize=(8, 5))
#     hist, bins, _ = ax.hist(norms, bins=10, color='#607c8e', alpha=0.25)
#     ax.set_yscale('log10')
#
#     for i, threshold in enumerate(thresholds):
#         ax.axvline(x=threshold, color=colors[i], linestyle='--', label=f'Threshold = {threshold} mm')
#
#     for i, count in enumerate(hist):
#         ax.text(bins[i], count + 5, str(int(count)), ha='left', va='bottom', fontsize=10)
#
#     set_ax_info(
#         ax,
#         xlabel=f'{x_label}',
#         ylabel='Accuracy',
#         title=f'Distribution of Prediction Errors'
#     )
#
#     fig.tight_layout()
#     fig.savefig(f'plots/histogram_{name}.pdf')
#     plt.close(fig)


# def make_histogram(num_bins, norms, thresholds, x_label, dist, name):
#     color = 'steelblue'  # Choose a color for the histogram bars
#     colors = ['r', 'g', 'b', 'm']
#
#     fig, ax = plt.subplots(figsize=(8, 5))
#
#     hist, bins, _ = ax.hist(norms, bins=np.linspace(0, num_bins, num_bins + 1), color=color, alpha=0.7, rwidth=0.5)  # Add rwidth parameter
#     ax.set_yscale('log')
#
#     for i, count in enumerate(hist):
#         ax.text(bins[i], count + 5, str(int(count)), ha='left', va='bottom', fontsize=10)
#
#     ax.set_xlim(0, num_bins)  # Set the x-axis limits
#     ax.set_xticks(np.arange(0, num_bins + 1, dist))  # Set the x-axis ticks at 2, 4, 6, 8, ...
#     set_ax_info(
#         ax,
#         xlabel=f'{x_label}',
#         ylabel='Accuracy',
#         title=f'Distribution of Prediction Errors'
#     )
#
#     fig.tight_layout()
#     fig.savefig(f'plots/new_histogram_{name}.pdf')
#     plt.close(fig)

def make_histogram(num_bins, norms, thresholds, x_label, dist, name):
    color = 'steelblue'  # Choose a color for the histogram bars
    colors = ['r', 'g', 'b', 'm']

    total_bins = num_bins // 3  # Calculate the total number of bins

    fig, ax = plt.subplots(figsize=(8, 5))

    hist, bins, _ = ax.hist(norms, bins=np.linspace(0, num_bins, total_bins + 1), color=color, alpha=0.7, rwidth=0.5)  # Update the number of bins
    ax.set_yscale('log')

    label_offset = 0.1
    for i, count in enumerate(hist):
        # ax.text(bins[i], count + 5, str(int(count)), ha='left', va='bottom', fontsize=10)
        ax.text((bins[i] + bins[i + 1]) / 2 + label_offset, count + 5, str(int(count)), ha='center', va='bottom', fontsize=10)

    ax.set_xlim(0, num_bins)  # Set the x-axis limits
    dist = num_bins / total_bins  # Calculate the distance between x-axis ticks
    ax.set_xticks(np.arange(0, num_bins + 1, dist))  # Set the x-axis ticks
    set_ax_info(
            ax,
            xlabel=f'{x_label}',
            ylabel='Number of Samples',
            title=f'Distribution of Prediction Errors'
        )

    fig.tight_layout()
    fig.savefig(f'plots/new_histogram_{name}.pdf')
    plt.close(fig)

def one_sample(predicted, target):
    N_dipoles = 2
    N_outputs = len(predicted) // N_dipoles
    combinations = itertools.permutations(range(N_dipoles), N_dipoles) # [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    result = 1e99

    individual_dist = []

    for pred_indices in combinations:
        tmp = 0
        for i, j in enumerate(pred_indices):
            euc_dist = one_dipole(
                predicted[N_outputs*j:N_outputs*(j + 1)],
                target[N_outputs*i:N_outputs*(i + 1)]
            )
            individual_dist.append(euc_dist)
            tmp += euc_dist

        result = min(result, tmp)

    sum_1 = individual_dist[0] + individual_dist[1]
    sum_2 = individual_dist[2] + individual_dist[3]

    if sum_1 > sum_2:
        individual_dist.remove(individual_dist[0])
        individual_dist.remove(individual_dist[0])
    else:
        individual_dist.remove(individual_dist[-1])
        individual_dist.remove(individual_dist[-1])

    return result, individual_dist

def one_dipole(predicted, target):
    euc_dist = np.linalg.norm(predicted[:3] - target[:3])
    return euc_dist

def test_criterea_two_dipoles(predictions, targets, name):
    num_samples = predictions.shape[0]
    result = np.zeros(num_samples)
    norms_1 = np.zeros(num_samples)
    norms_2 = np.zeros(num_samples)
    for i in range(num_samples):
        result_one_sample , euc_dist = one_sample(predictions[i], targets[i])
        result[i] = result_one_sample
        norms_1[i] = euc_dist[0]
        norms_2[i] = euc_dist[1]


    print(f'Mean Euclidean Distance (MED) for dipoles is {np.mean(result)}')
    print(f'Mean Euclidean Distance (MED) for dipole 1 is {np.mean(norms_1)}')
    print(f'Mean Euclidean Distance (MED) for dipole 2 is {np.mean(norms_2)}')

    norms_list = [norms_1, norms_2]

    thresholds = [5, 10, 15]
    for norms in (norms_list):
        results = np.zeros(3)
        for i, threshold in enumerate(thresholds):
            results[i] = sum(norms < threshold)

        results = results/num_samples*100

        l = [
            ['', 'MED < 5 mm', 'MED < 10 mm', 'MED < 15 mm'],
            ['pos', f'{results[0]:.3f}',
                    f'{results[1]:.3f}',
                    f'{results[2]:.3f}']
        ]
        table = Table(l)
        table.write()


    result_averaged_over_dipoles = result/2
    threshold_results = np.zeros(3)
    for i, threshold in enumerate(thresholds):
        threshold_results[i] = sum(result_averaged_over_dipoles < threshold)

    threshold_results = threshold_results/num_samples*100

    l = [
        ['', 'MED < 5 mm', 'MED < 10 mm', 'MED < 15 mm'],
        ['pos', f'{threshold_results[0]:.3f}',
                f'{threshold_results[1]:.3f}',
                f'{threshold_results[2]:.3f}']
    ]
    table = Table(l)
    table.write()

    make_histogram(45, result_averaged_over_dipoles, thresholds, 'Euclidean Distance [mm]', 3, f'2_dipoles_position_{name}')



def test_criterea(predictions, targets, name):
    if np.shape(predictions)[1] == 6:
        test_criterea_two_dipoles(predictions, targets, name)

    else:
        norms = np.linalg.norm(predictions[:,:3] - targets[:,:3], axis=1)

        print(f'Mean Euclidean Distance (MED) is {np.mean(norms)}')

        thresholds = [5, 10, 15]
        results = np.zeros(4)
        for i, threshold in enumerate(thresholds):
            results[i] = sum(norms < threshold)

        results = results/np.shape(predictions)[0]*100

        l = [
            ['', 'MED < 5 mm', 'MED < 10 mm', 'MED < 15 mm'],
            ['pos', f'{results[0]:.3f}',
                    f'{results[1]:.3f}',
                    f'{results[2]:.3f}']
        ]
        table = Table(l)
        table.write()

        make_histogram(26, norms, thresholds, 'Euclidean Distance [mm]', 2, f'position_{name}')

        if name == 'amplitude' or name == 'area':
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

            make_histogram(10, amplitude_absolute_error, thresholds, r'Amplitude Absolute Error [mA$\mu$m]', 2, f'amplitude_{name}')

            results_combined = np.zeros(len(thresholds))
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

            # make_histogram(10, results_combined, thresholds, 'Amplitude Absolute Error (mm)', name)

        if name == 'area':
            area_absolute_error = np.abs(predictions[:,4] - targets[:,4])
            print(f'Mean Absolute Error (MAE) for area is {np.mean(area_absolute_error)}')

            amplitude_absolute_error = np.abs(predictions[:,3] - targets[:,3])

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

            make_histogram(10, area_absolute_error, thresholds, 'Area Absolute Error (mm)', 2, f'area_{name}')

            results_combined = np.zeros(3)
            for i, threshold in enumerate(thresholds):
                results_combined[i] = sum((norms < 10) & (amplitude_absolute_error < 1) & (area_absolute_error < threshold))

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
