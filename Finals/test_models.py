import json

import numpy as np
from ft.formats import Table

from utils import MSE, MAE


class TestResults:
    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self._predictions = predictions.copy()
        self._targets = targets.copy()

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


def generate_test_results(predictions: np.ndarray, targets: np.ndarray):
    results = TestResults(predictions, targets)
    d = {
        'MAE_x': results.MAE_x,
        'MAE_y': results.MAE_y,
        'MAE_z': results.MAE_z,
        'MAE_amplitude': results.MAE_amplitude,
        'MAE_radius': results.MAE_radius,
        'MAE_position': results.MAE_position,
        'MSE_x': results.MSE_x,
        'MSE_y': results.MSE_y,
        'MSE_z': results.MSE_z,
        'MSE_amplitude': results.MSE_amplitude,
        'MSE_radius': results.MSE_radius,
        'MSE_position': results.MSE_position
    }
    return d


def print_test_results(d):
    l = [
        ['', 'MAE (mm)', 'MSE (mm^2)', 'RMSE (mm)'],
        ['x', f'{d["MAE_x"]:.3f}', f'{d["MSE_x"]:.3f}', f'{np.sqrt(d["MSE_x"]):.3f}'],
        ['y', f'{d["MAE_y"]:.3f}', f'{d["MSE_y"]:.3f}', f'{np.sqrt(d["MSE_y"]):.3f}'],
        ['z', f'{d["MAE_z"]:.3f}', f'{d["MSE_z"]:.3f}', f'{np.sqrt(d["MSE_z"]):.3f}'],
        ['Position', f'{d["MAE_position"]:.3f}', f'{d["MSE_position"]:.3f}', f'{np.sqrt(d["MSE_position"]):.3f}'],
        ['Amplitude', f'{d["MAE_amplitude"]:.3f}', f'{d["MSE_amplitude"]:.3f}', f'{np.sqrt(d["MSE_amplitude"]):.3f}'],
        ['Radius', f'{d["MAE_radius"]:.3f}', f'{d["MSE_radius"]:.3f}', f'{np.sqrt(d["MSE_radius"]):.3f}'],
    ]
    table = Table(l)
    table.write()


def save_test_results(d, filename: str):
    with open(filename, 'w', encoding='UTF-8') as f:
        json.dump(d, f, indent=2)


def load_test_results(filename: str):
    with open(filename, 'r', encoding='UTF-8') as f:
        return json.load(f)
