import json

import numpy as np

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
    def MAE_radius(self):
        return self._MAE(3)

    @property
    def MAE_amplitude(self):
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
    def MSE_radius(self):
        return self._MSE(3)

    @property
    def MSE_amplitude(self):
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
        'MAE_radius': results.MAE_radius,
        'MAE_amplitude': results.MAE_amplitude,
        'MAE_position': results.MAE_position,
        'MSE_x': results.MSE_x,
        'MSE_y': results.MSE_y,
        'MSE_z': results.MSE_z,
        'MSE_radius': results.MSE_radius,
        'MSE_amplitude': results.MSE_amplitude,
        'MSE_position': results.MSE_position
    }
    return d


def print_test_results(d):
    s = '' \
        + f'MAE x-coordinates:{d["MAE_x"]} mm\n' \
        + f'MAE y-coordinates:{d["MAE_y"]} mm\n' \
        + f'MAE z-coordinates:{d["MAE_z"]} mm\n' \
        + f'MAE radius:{d["MAE_radius"]} mm\n' \
        + f'MAE amplitude:{d["MAE_amplitude"]} mm\n' \
        + f'MAE: {d["MAE_position"]} mm\n' \
        + '\n' \
        + f'MSE x-coordinates:{d["MSE_x"]} mm\n' \
        + f'MSE y-coordinates:{d["MSE_y"]} mm\n' \
        + f'MSE y-coordinates:{d["MSE_z"]} mm\n' \
        + f'MSE radius:{d["MSE_radius"]} mm\n' \
        + f'MSE amplitude:{d["MSE_amplitude"]} mm\n' \
        + f'MSE: {d["MSE_position"]} mm\n' \
        + '\n' \
        + f'RMSE x-coordinates:{np.sqrt(d["MSE_x"])}\n' \
        + f'RMSE y-coordinates:{np.sqrt(d["MSE_y"])}\n' \
        + f'RMSE z-coordinates:{np.sqrt(d["MSE_z"])}\n' \
        + f'RMSE radius:{np.sqrt(d["MSE_radius"])}\n' \
        + f'RMSE amplitude:{np.sqrt(d["MSE_amplitude"])}\n' \
        + f'RMSE: {np.sqrt(d["MSE_position"])} mm'
    print(s)


def save_test_results(d, filename: str):
    with open(filename, 'w', encoding='UTF-8') as f:
        json.dump(d, f, indent=2)


def load_test_results(filename: str):
    with open(filename, 'r', encoding='UTF-8') as f:
        return json.load(f)
