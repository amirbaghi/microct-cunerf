import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class MLPTransformerRHINO(nn.Module):
    def __init__(self, input_dim = 3, num_layers = 1, hidden_dim = 64, freq = 9):
        super().__init__()
        self.input_dim = input_dim * freq * 2
        self.output_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.freq = freq

        backbone = []

        # Add the in and out layer + the hidden layers inbetween.
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim
            
            # Bias set to true, False might be better.
            backbone.append(nn.Linear(in_dim, out_dim, bias=True))

        self.backbone = nn.ModuleList(backbone)

    def gamma(self, x):

        # x = x.unsqueeze(-1) # TODO: may need to remove.
        # Create a tensor of powers of 2
        scales = 2.0 ** torch.arange(self.freq)
        # Compute sin and cos features
        features = torch.cat([torch.sin(x * np.pi * scale) for scale in scales] + [torch.cos(x * np.pi * scale) for scale in scales], dim=-1)
        return features

    def forward(self, x):
        
        g = self.gamma(x)

        for l in range(self.num_layers):
            g = self.backbone[l](g)
            if l != self.num_layers - 1:
                # If not last layer, apply ReLU (0+)
                g = F.relu(g, inplace=False)
        return g