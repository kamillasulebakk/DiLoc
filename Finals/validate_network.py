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
    name = 'amplitude'
    data = EEGDataset('test', parameters)
    print(model(data.eeg))
    print(data.target)
    predictions =  data.denormalize(model(data.eeg)).detach().numpy()
    targets = data.denormalize(data.target).detach().numpy()
    print(predictions)
    print(targets)
    input()
    # if name == 'amplitude':
    #     plot_error_amplitude(predictions, targets)

    criterea = tm.test_criterea(predictions, targets, name)
    test_results = tm.generate_test_results(predictions, targets)
    tm.print_test_results(test_results)
    tm.save_test_results(test_results, name)

def main():
    # Simple
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.1_0.0_500_(0).pt')
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.5_0.0_500_(0).pt')









    # Amplitude
    # NB, weight = '1, 1'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(0).pt')

    # NB, weight = '1, 1'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(1).pt')

    # NB, weigh = '0.3, 0.7'
    # model = torch.load('trained_models/amplitudes_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(7).pt')

    # NB, weight = '1, 2'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(2).pt')

    # model = torch.load('trained_models/amplitudes_new_custom_loss_tanh_32_0.001_0.35_0.1_0_2000_(1).pt')
    # model = torch.load('trained_models/amplitudes_new_custom_loss_tanh_32_0.001_0.35_0.1_0_2000_(0).pt')














    # Two dipoles
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.5_0_6000_(1).pt')
    # model = torch.load('trained_models/amplitudes_64_0.001_0.35_0.1_1e-05_3000_(1).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0_10000_(0).pt')
    # model = torch.load('trained_models/amplitudes_32_0.001_0.35_0.1_0_6000_(3).pt')


    # model = torch.load('trained_models/amplitudes_custom_loss_32_0.001_0.35_0.1_0_8000_(0).pt')
    """
    Mean Euclidean Distance (MED) is 47.496925354003906
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    1.290     |    4.850     |    19.145     |    31.235     |
    -----------------------------------------------------------------------
    """

    model = torch.load('trained_models/simple_custom_loss_constA_ReLU_32_0.001_0.35_0.1_0_1500_(0).pt')
    """
    Mean Euclidean Distance (MED) is 44.97166061401367
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    6.700     |    19.240    |    40.185     |    46.790     |
    -----------------------------------------------------------------------
    """


    # model = torch.load('trained_models/simple_custom_loss_constA_32_0.001_0.35_0.1_0_1500_(0).pt')
    """
    Mean Euclidean Distance (MED) is 44.69486999511719
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    4.540     |    14.775    |    37.380     |    45.755     |
    -----------------------------------------------------------------------
    """
    # model = torch.load('trained_models/simple_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(1).pt')
    """
    Mean Euclidean Distance (MED) is 46.394039154052734
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    2.310     |    8.480     |    26.375     |    37.670     |
    -----------------------------------------------------------------------
    """

    # model = torch.load('trained_models/simple_custom_loss_relu_32_0.001_0.35_0.1_0_1500_(1).pt')
    """
    Mean Euclidean Distance (MED) is 47.358367919921875
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    1.050     |    4.180     |    18.710     |    31.830     |
    -----------------------------------------------------------------------
    """


    # model = torch.load('trained_models/simple_custom_loss_tanh_32_0.001_0.35_0.1_0_2000_(1).pt')
    # model = torch.load('trained_models/simplellrelu_custom_loss_relu_32_0.001_0.35_0.1_0_1500_(0).pt')





    # Area
    # model = torch.load('trained_models/area_32_0.001_0.35_0.1_0.0_5000_(0).pt')
    # model = torch.load('trained_models/area_custom_loss_ReLU_32_0.001_0.35_0.1_0_1500_(1).pt')
    #OMG:
    # model = torch.load('trained_models/area_custom_loss_32_0.001_0.35_0.1_0_6000_(0).pt')




    print('Pretrained model loaded')
    validate_network(model, amplitude.basic_parameters())

if __name__ == '__main__':
    main()
