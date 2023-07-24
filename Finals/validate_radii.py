import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN

from utils import numpy_to_torch, denormalize
import test_models as tm


# eeg = np.load('data/validate_const_A_dipole_area_const_A_eeg_70000_1.npy')
# eeg = (eeg - np.mean(eeg))/np.std(eeg)
# eeg = numpy_to_torch(eeg)
# target = np.load('data/validate_const_A_dipole_area_const_A_locations_70000_1.npy')
# print('Test data loaded')

# amplitude
# eeg = np.load('data/amplitudes_70000_1_eeg_test.npy')
# eeg = (eeg - np.mean(eeg))/np.std(eeg)
# eeg = numpy_to_torch(eeg)
# target = np.load('data/amplitudes_70000_1_targets_test.npy')
# print('Test data loaded')
#
# model = torch.load('trained_models/july/amplitudes_32_0.001_0.35_0.1_(5).pt')

# simple
eeg = np.load('data/simple_70000_1_eeg_test.npy')
eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg)
target = np.load('data/simple_70000_1_targets_test.npy')
print('Test data loaded')

model = torch.load('trained_models/july/simple_64_0.001_0.35_0.1_(2).pt')


print('Pretrained model loaded')

predictions = model(eeg).detach().numpy()

x_target = target[:, 0]
y_target = target[:, 1]
z_target = target[:, 2]
# amplitude_target = target[:, -2]
# radius_target = target[:, -1]

max_targets = np.array([
    72.02555727958679,
    73.47751750051975,
    81.150386095047,
    10,
    15
])
min_targets = np.array([
    -72.02555727958679,
    -106.12010800838469,
    -52.66008937358856,
    1,
    0
])

# predictions_denormalized = np.zeros_like(predictions)
# for i in range(predictions.shape[1]):
#     predictions_denormalized[:, i] = denormalize(predictions[:, i], max_targets[i], min_targets[i])

# test_results = tm.generate_test_results(predictions_denormalized, target)
test_results = tm.generate_test_results(predictions, target)
tm.print_test_results(test_results)
tm.save_test_results(test_results, filename='simple')
