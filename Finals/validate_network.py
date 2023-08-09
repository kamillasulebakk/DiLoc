import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN
from eeg_dataset import EEGDataset

from utils import numpy_to_torch, denormalize
from investigate_amplitude import plot_error_amplitude
import test_models as tm
import simple
import amplitude
import area


def validate_network(model, parameters):
    name = 'area'
    data = EEGDataset('test', parameters)
    predictions =  data.denormalize(model(data.eeg)).detach().numpy()
    targets = data.denormalize(data.target).detach().numpy()

    norms = np.linalg.norm(predictions[:,:3] - targets[:,:3], axis=1)
    # within_threshold = sum(norms < 10)

    amplitude_absolute_error = np.abs(predictions[:,3] - targets[:,3])
    within_threshold = sum((norms < 10) & (amplitude_absolute_error < 0.6))


    if name == 'amplitude':
        plot_error_amplitude(predictions, targets)

    test_results = tm.generate_test_results(predictions, targets)
    tm.print_test_results(test_results)
    tm.save_test_results(test_results, name)

def main():
    # Simple
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.1_0.0_500_(0).pt')
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.5_0.0_500_(0).pt')

    # Amplitude
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0.0_3000_(0).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0_6000_(0).pt')

    # Two dipoles
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.5_0_6000_(1).pt')
    # model = torch.load('trained_models/amplitudes_64_0.001_0.35_0.1_1e-05_3000_(1).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.5_0_6000_(2).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0_10000_(0).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0_6000_(3).pt')

    # Area
    # model = torch.load('trained_models/area_32_0.001_0.35_0.1_0_10000_(0).pt')
    model = torch.load('trained_models/area_32_0.001_0.35_0.1_0.0_5000_(0).pt')
    # model = torch.load('trained_models/area_64_0.001_0.35_0.1_0_10000_(0).pt')
    # model = torch.load('trained_models/area_64_0.001_0.35_0.5_1e-05_5000_(0).pt')

    print('Pretrained model loaded')
    validate_network(model, area.basic_parameters())

if __name__ == '__main__':
    main()

# # area
# eeg = np.load('data/area_70000_1_eeg_test.npy')
# eeg = (eeg - np.mean(eeg))/np.std(eeg)
# eeg = numpy_to_torch(eeg)
# target = np.load('data/area_70000_1_targets_test.npy')
# print('Test data loaded')
# model = torch.load('trained_models/area_32_0.001_0.35_0.1_0.0_5000_(0).pt')
#
#
# # # amplitudes
# # eeg = np.load('data/amplitudes_70000_1_eeg_test.npy')
# # eeg = (eeg - np.mean(eeg))/np.std(eeg)
# # eeg = numpy_to_torch(eeg)
# # target = np.load('data/amplitudes_70000_1_targets_test.npy')
# # print('Test data loaded')
# # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0.0_3000_(0).pt')
#
# # # simple
# # eeg = np.load('data/simple_70000_1_eeg_test.npy')
# # eeg = (eeg - np.mean(eeg))/np.std(eeg)
# # eeg = numpy_to_torch(eeg)
# # target = np.load('data/simple_70000_1_targets_test.npy')
# # print('Test data loaded')
# # model = torch.load('trained_models/simple_32_0.001_0.35_0.1_0.0_500_(0).pt')
#
#
# print('Pretrained model loaded')
#
# predictions = model(eeg).detach().numpy()
#
# x_target = target[:, 0]
# y_target = target[:, 1]
# z_target = target[:, 2]
# amplitude_target = target[:, -2]
# # radius_target = target[:, -1]
#
# max_targets = np.array([
#     72.02555727958679,
#     73.47751750051975,
#     81.150386095047,
#     10,
#     15
# ])
# min_targets = np.array([
#     -72.02555727958679,
#     -106.12010800838469,
#     -52.66008937358856,
#     1,
#     0
# ])
#
# predictions_denormalized = np.zeros_like(predictions)
# for i in range(predictions.shape[1]):
#     predictions_denormalized[:, i] = denormalize(
    # predictions[:, i], max_targets[i], min_targets[i])
#
# test_results = tm.generate_test_results(predictions_denormalized, target)
# # test_results = tm.generate_test_results(predictions, target)
# tm.print_test_results(test_results)
# tm.save_test_results(test_results, filename='area')
