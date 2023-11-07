# DiLoc
Master's thesis. Machine learning tools for localization and identification of dipoles from EEG signals.

## Code Implementation
In this work we have utilized the New York Head Model implemented in the Python module LFPy 2.0, to simulate EEG data. Neural Networks has been build using Pytorch. 
### The project is created with:
* Python version: 3.7.6
  * Pytorch 1.13.0
  * h5py 2.10.0
  * LFPy 2.0
  
* LaTeX

## Data 
All data used in the thesis can be found under Finals/data. To produce new data 

## How to train the networks:
```
python simple.py
python amplitude.py
python area.py
python simple_cnn.py

```
Here; 
* python simple.py trains the neural network for localizing single dipoles. Hyperparameters can be adjusted in this file using a FCNN. 
* python amplitude.py trains the neural network for localizing single and pair of dipoles with and without varying amplitudes using a FCNN. Hyperparameters can be adjusted in this file. 
* python area.py trains the neural network for localizing spherical populations of dipoles using a FCNN. Hyperparameters can be adjusted in this file. 
* python simple_cnn.py trains the neural network for localizing single and pair of dipoles using a CNN. Hyperparameters can be adjusted in this file. 


## How to test the performance of the networks presented in the final report:
```
python validate_networks_in_thesis.py

```
* The choice of which network to test must be adjusted in this file. 

