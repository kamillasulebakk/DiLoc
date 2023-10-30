import torch
import numpy as np
import matplotlib.pyplot as plt

from ffnn import FFNN
from eeg_dataset import EEGDataset

from utils import numpy_to_torch #, denormalize
from investigate_amplitude import plot_error_amplitude, plot_error_area
import test_models as tm
import simple
import amplitude
import area
import simple_cnn

# palette = sns.color_palette("deep")
# sns.set_palette(palette)

def validate_network(model, parameters):
    name = 'simple_cnn'
    data = EEGDataset('test', parameters)

    if name == 'simple':
        predictions =  model(data.eeg).detach().numpy()
        targets = data.target.detach().numpy()

    elif name == 'simple_cnn':
        eeg = data.eeg
        predictions = model(eeg.unsqueeze(1)).detach().numpy()
        print(predictions)
        input()
        targets = (data.target).detach().numpy()
    else:
        predictions =  data.denormalize(model(data.eeg)).detach().numpy()
        targets = data.denormalize(data.target).detach().numpy()

    if name == 'amplitude':
        plot_error_amplitude(predictions, targets)
    if name == 'area':
        plot_error_area(predictions, targets)

    criterea = tm.test_criterea(predictions, targets, name)
    test_results = tm.generate_test_results(predictions, targets)
    tm.print_test_results(test_results)
    tm.save_test_results(test_results, name)




def main():
    # Simple
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.1_0.0_500_(0).pt')
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.5_0.0_500_(0).pt')
    # model = torch.load('trained_models/simple_new_standarization_tanh_32_0.001_0.35_0.5_0_500_(5).pt')
    # model = torch.load('trained_models/simple_new_new_standarization_tanh_32_0.001_0.35_0.5_0_500_(0).pt')
    # model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.1_0_1500_(0).pt')


    # Amplitude
    # NB, weight = '1, 1'
    model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(0).pt')


    # Two dipoles
    # model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.1_0_800_(2).pt')
    # model = torch.load('trained_models/simple_last_run_old_std_2_dipoles_32_0.001_0.35_0.1_0_800_(0).pt') # right one
    # model = torch.load('trained_models/simple_last_run_old_std_2_dipoles_32_0.001_0.35_0.5_0_800_(0).pt') # right one


    # Area
    # model = torch.load('trained_models/area_last_run_old_std_area_32_0.001_0.35_0.1_0_800_(1).pt')
    # model = torch.load('trained_models/area_seed_42_cnn_32_0.001_0.35_0.1_0_1500_(0).pt')
    # model = torch.load('trained_models/area_seed_42_cnn_32_0.001_0.35_0.1_0_1000_(1).pt') # GOOD



    #CNN
    # model = torch.load('trained_models/simple_last_run_old_std_area_32_0.001_0.35_0.5_0_800_(0).pt')
    # model = torch.load('trained_models/simple_last_run_old_std_area_32_0.001_0.35_0.1_0_800_(0).pt')
    # model = torch.load('trained_models/simple_last_run_old_std_cnn_32_0.001_0.35_0.5_0_600_(0).pt')

    model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.5_0_800_(1).pt')
    # model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.1_0_800_(1).pt')

    # model = torch.load('trained_models/simple_last_run_old_std_2_dipoles_32_0.001_0.35_0.5_0_800_(5).pt')

    # model = torch.load('trained_models/simple_last_run_old_std_area_32_0.001_0.35_0.5_0_800_(0).pt')
    # model = torch.load('trained_models/simple_last_run_old_std_area_32_0.001_0.35_0.1_0_800_(0).pt')




    #two dipoles:
    # model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.1_0_800_(1).pt')
    # model = torch.load('trained_models/simple_seed_42_cnn_32_0.001_0.35_0.5_0_800_(1).pt') # GOOD

    '''
    Seems like result varies a bit from seed to seed. Maybe it is not necessary to give which seed has been used.
    '''


    print('Pretrained model loaded')
    validate_network(model, simple_cnn.basic_parameters())

if __name__ == '__main__':
    main()
