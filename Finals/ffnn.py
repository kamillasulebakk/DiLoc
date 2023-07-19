from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class FFNN(nn.Module):
    def __init__(
        self,
        hidden_layers: List[int],
        N_dipoles: int,
        determine_area: bool = False,
        determine_amplitude: bool = False
    ):
        super().__init__()
        self.determine_area = determine_area
        self.determine_amplitude = determine_amplitude
        self.dropout = nn.Dropout(p=0.5)

        self.first_layer = nn.Linear(231, hidden_layers[0])
        self.hidden_layers = []
        for i in range(len(hidden_layers) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_layers[i], hidden_layers[i+1])
            )
        number_of_output_values = 3
        if determine_area:
            number_of_output_values += 1
        if determine_amplitude:
            number_of_output_values += 1
        self.final_layer = nn.Linear(
            hidden_layers[-1],
            number_of_output_values*N_dipoles
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.first_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.final_layer(x)
        if self.determine_area or self.determine_amplitude:
            x = torch.sigmoid(x)
        return x