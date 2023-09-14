from typing import List

import torch
from torch import nn


def number_of_output_values(amp: bool, area: bool):
    result = 3
    if amp:
        result += 1
    if area:
        result += 1
    return result


class FFNN(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        hidden_layers: List[int] = parameters['hidden_layers']
        N_dipoles: int = parameters['N_dipoles']
        self.determine_area = parameters['determine_area']
        self.determine_amplitude = parameters['determine_amplitude']
        self.hl_activation_function = parameters['hl_act_func']
        self.dropout = nn.Dropout(p=0.5)

        self.first_layer = nn.Linear(231, hidden_layers[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i+1])
            )
        N_output = number_of_output_values(
            self.determine_amplitude, self.determine_area
        )
        self.final_layer = nn.Linear(
            hidden_layers[-1],
            N_output*N_dipoles
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor):
        x = nn.functional.relu(self.first_layer(x))
        # x = torch.tanh(self.first_layer(x))

        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
            # if self.hl_activation_function == 'tanh':
            #     x = torch.tanh(layer(x))
            # else:
            #     x = nn.functional.relu(layer(x))
        x = self.final_layer(x)
        # x = torch.sigmoid(x)

        # if self.determine_area or self.determine_amplitude:
        #     x = torch.sigmoid(x)
        return x
