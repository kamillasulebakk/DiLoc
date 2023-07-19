import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN

from utils import numpy_to_torch, denormalize
import test_models as tm

# fig, ax = plt.subplots()
# ax.plot(x, y)
# set_ax_info(ax, 'Amplitude [mm]', 'Error [mm]', 'Relative difference', legend=False)
# fig.tight_layout()
# fig.savefig('plots/dipole_w_radi_amplitude_mse_lr_1.5.png')

# eeg, target = load_data_files(50000, 'dipole_area', num_dipoles=1)

# eeg = np.load('data/validate_dipole_area_eeg_70000.npy')
# target = np.load('data/validate_dipole_area_locations_70000.npy')
eeg = np.load('data/validate_const_A_dipole_area_const_A_eeg_70000_1.npy')
eeg = (eeg - np.mean(eeg))/np.std(eeg)
eeg = numpy_to_torch(eeg)
target = np.load('data/validate_const_A_dipole_area_const_A_locations_70000_1.npy')
print('Test data loaded')

# model = torch.load('trained_models/22.juni_dipole_w_radi_amplitude_500_SGD_lr0.09_wd0.0001_mom0.9_bs32.pt')
# model = torch.load('trained_models/22.juni_dipole_w_radi_amplitude_500_SGD_lr0.9_wd0.0001_mom0.9_bs32.pt')
# model = torch.load('trained_models/50000_26junemseloss_MSE_area_w_amplitude_5000_SGD_lr0.001_wd0.1_mom0.35_bs64.pt')
# model = torch.load('trained_models/july/new_dataset_simple_network_radius_tanh_sigmoid_50000_12july_mseloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.1_bs32.pt')
# model = torch.load('trained_models/july/l1_0.0001_simple_network_radius_tanh_sigmoid_50000_12july_mseloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.1_bs32.pt')
# model = torch.load('trained_models/july/l1_0.1_simple_network_radius_tanh_sigmoid_50000_12july_mseloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.1_bs32.pt')
#
# model = torch.load('trained_models/july/area_l1_and_l2_less_complicated_network_radius_tanh_sigmoid_50000_18july_mseloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.05_bs32.pt')
# model = torch.load('trained_models/july/area_l2_less_complicated_network_radius_tanh_sigmoid_50000_18july_mseloss_MSE_dipole_w_amplitude_3000_SGD_lr0.001_mom0.35_wd_0.05_bs32.pt')

model = torch.load('trained_models/july/test123.pt')

print('Pretrained model loaded')

predictions = model(eeg).detach().numpy()

x_target = target[:, 0]
y_target = target[:, 1]
z_target = target[:, 2]
radius_target = target[:, -2]
amplitude_target = target[:, -1]

predictions_denormalized = np.zeros_like(predictions)
for i in range(predictions.shape[1]):
    predictions_denormalized[:, i] = denormalize(predictions[:, i], np.max(target[:, i]), np.min(target[:, i]))

test_results = tm.generate_test_results(predictions_denormalized, target)
tm.print_test_results(test_results)
tm.save_test_results(test_results, filename='bladibla')
