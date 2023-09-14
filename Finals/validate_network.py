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
    name = 'simple'
    data = EEGDataset('test', parameters)
    predictions =  data.denormalize(model(data.eeg)).detach().numpy()
    targets = data.denormalize(data.target).detach().numpy()
    # if name == 'amplitude':
    #     plot_error_amplitude(predictions, targets)

    criterea = tm.test_criterea(predictions, targets, name)
    test_results = tm.generate_test_results(predictions, targets)
    tm.print_test_results(test_results)
    tm.save_test_results(test_results, name)

def main():
    # Simple
    # model = torch.load('trained_models/simple_32_0.001_0.35_0.1_0.0_500_(0).pt')
    model = torch.load('trained_models/simple_32_0.001_0.35_0.5_0.0_500_(0).pt')
    # model = torch.load('trained_models/simple_new_standarization_tanh_32_0.001_0.35_0.1_0_2000_(0).pt')
    # model = torch.load('trained_models/simple_new_standarization_tanh_32_0.001_0.35_0.5_0_500_(5).pt')
    # model = torch.load('trained_models/simple_new_new_standarization_tanh_32_0.001_0.35_0.5_0_500_(0).pt')







    # Amplitude
    # NB, weight = '1, 1'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(0).pt')
    """
    Mean Euclidean Distance (MED) is 2.805983781814575
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    64.325    |    90.930    |    99.505     |    99.925     |
    -----------------------------------------------------------------------
    No handles with labels found to put in legend.
    Mean Absolute Error (MAE) for amplitude is 0.5386642813682556
    ---------------------------------------------------------------------------------
    |             |  MAE < 1.0 mA$mu$m  |  MAE < 2.0 mA$mu$m  |  MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------|
    |  amplitude  |       85.070        |       96.870        |       99.220        |
    ---------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------
    |                   |  MED < 10 mm & MAE < 1.0 mA$mu$m  |  MED < 10 mm & MAE < 2.0 mA$mu$m  |  MED < 10 mm & MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------------------------------------------------------|
    |  pos & amplitude  |              84.645               |              96.410               |              98.740               |
    ---------------------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------
    |             |  MAE (mm)  |  MSE (mm^2)  |  RMSE (mm)  |
    |-------------------------------------------------------|
    |      x      |   1.348    |    3.438     |    1.854    |
    |      y      |   1.448    |    3.860     |    1.965    |
    |      z      |   1.420    |    3.862     |    1.965    |
    |  Position   |   1.405    |    3.720     |    1.929    |
    |  Amplitude  |   0.539    |    0.650     |    0.806    |
    ---------------------------------------------------------
    """

    # NB, weight = '1, 1'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(1).pt')
    """
    Mean Euclidean Distance (MED) is 2.853764533996582
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    63.330    |    89.990    |    99.540     |    99.915     |
    -----------------------------------------------------------------------
    No handles with labels found to put in legend.
    Mean Absolute Error (MAE) for amplitude is 0.545149028301239
    ---------------------------------------------------------------------------------
    |             |  MAE < 1.0 mA$mu$m  |  MAE < 2.0 mA$mu$m  |  MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------|
    |  amplitude  |       84.840        |       96.975        |       99.225        |
    ---------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------
    |                   |  MED < 10 mm & MAE < 1.0 mA$mu$m  |  MED < 10 mm & MAE < 2.0 mA$mu$m  |  MED < 10 mm & MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------------------------------------------------------|
    |  pos & amplitude  |              84.505               |              96.590               |              98.810               |
    ---------------------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------
    |             |  MAE (mm)  |  MSE (mm^2)  |  RMSE (mm)  |
    |-------------------------------------------------------|
    |      x      |   1.378    |    3.538     |    1.881    |
    |      y      |   1.498    |    4.214     |    2.053    |
    |      z      |   1.405    |    3.738     |    1.933    |
    |  Position   |   1.427    |    3.830     |    1.957    |
    |  Amplitude  |   0.545    |    0.658     |    0.811    |
    ---------------------------------------------------------
    """

    # NB, weigh = '0.3, 0.7'
    # model = torch.load('trained_models/amplitudes_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(7).pt')
    """
    Mean Euclidean Distance (MED) is 3.6360387802124023
    -----------------------------------------------------------------------
    |       |  MED < 3 mm  |  MED < 5 mm  |  MED < 10 mm  |  MED < 15 mm  |
    |---------------------------------------------------------------------|
    |  pos  |    44.965    |    80.500    |    98.605     |    99.740     |
    -----------------------------------------------------------------------
    No handles with labels found to put in legend.
    Mean Absolute Error (MAE) for amplitude is 0.5501509308815002
    ---------------------------------------------------------------------------------
    |             |  MAE < 1.0 mA$mu$m  |  MAE < 2.0 mA$mu$m  |  MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------|
    |  amplitude  |       84.665        |       96.905        |       99.145        |
    ---------------------------------------------------------------------------------
    ---------------------------------------------------------------------------------------------------------------------------------
    |                   |  MED < 10 mm & MAE < 1.0 mA$mu$m  |  MED < 10 mm & MAE < 2.0 mA$mu$m  |  MED < 10 mm & MAE < 3.0 mA$mu$m  |
    |-------------------------------------------------------------------------------------------------------------------------------|
    |  pos & amplitude  |              83.565               |              95.640               |              97.835               |
    ---------------------------------------------------------------------------------------------------------------------------------
    ---------------------------------------------------------
    |             |  MAE (mm)  |  MSE (mm^2)  |  RMSE (mm)  |
    |-------------------------------------------------------|
    |      x      |   1.714    |    5.306     |    2.304    |
    |      y      |   1.941    |    7.096     |    2.664    |
    |      z      |   1.791    |    6.140     |    2.478    |
    |  Position   |   1.815    |    6.181     |    2.486    |
    |  Amplitude  |   0.550    |    0.673     |    0.820    |
    ---------------------------------------------------------
    """
    # NB, weight = '1, 2'
    # model = torch.load('trained_models/amplitudes_test_custom_loss_tanh_32_0.001_0.35_0.1_0_1500_(2).pt')

    # model = torch.load('trained_models/amplitudes_new_custom_loss_tanh_32_0.001_0.35_0.1_0_2000_(1).pt')
    # model = torch.load('trained_models/amplitudes_new_custom_loss_tanh_32_0.001_0.35_0.1_0_2000_(0).pt')


    # NB, tanh all the way
    # model = torch.load('trained_models/amplitudes_new_standarization_tanh_32_0.001_0.35_0.1_0_3000_(1).pt')






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

    # model = torch.load('trained_models/simple_custom_loss_constA_ReLU_32_0.001_0.35_0.1_0_1500_(0).pt')
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
    validate_network(model, simple.basic_parameters())

if __name__ == '__main__':
    main()
