import torch
import torch.nn as nn
import torch.nn.functional as F

from transform_RHINO import MLPTransformerRHINO
from gridencoder import GridEncoder


class INGPNetworkRHINO(nn.Module):
    def __init__(self,
                # For backbone
                 encoding="hashgrid",
                 num_layers=5,
                 hidden_dim=128,
                 hidden_dim_last=128,
                #  For encoder
                 input_dim=3,
                 multires=6, 
                 degree=4,
                 num_levels=16,
                 level_dim=2,
                 base_resolution=16,
                 log2_hashmap_size=19,
                 desired_resolution=2048,
                 align_corners=False,
                # For transformer
                 freq = 9,
                 transformer_num_layers = 1,
                 transformer_hidden_dim = 64,
                 # Skips
                 skips = [4, 8]
                 ):
        super().__init__()

        self.encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)
        self.transformer = MLPTransformerRHINO(input_dim=3, num_layers=transformer_num_layers, hidden_dim=transformer_hidden_dim, freq=freq)
        self.in_dim = self.encoder.output_dim + self.transformer.output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.hidden_dim_last = hidden_dim_last
        self.skips = skips

        backbone = []

        # Add the in and out layer + the hidden layers inbetween.
        for l in range(num_layers - 1):
            if l == 0:
                in_dim = self.in_dim
            else:
                if l in self.skips:
                    in_dim = self.in_dim + self.hidden_dim
                    print("Indims: ", in_dim)
                else:
                    in_dim = self.hidden_dim
            
            if l == num_layers - 2:
                out_dim = self.hidden_dim_last
            else:
                out_dim = self.hidden_dim
            
            # Bias set to true, False might be better.
            backbone.append(nn.Linear(in_dim, out_dim, bias=True))

        backbone.append(nn.Linear(self.hidden_dim_last, 2, bias=True))

        self.backbone = nn.ModuleList(backbone)

    def forward(self, x):
        # Encode/hash x ... # x: [B, 3] numData/dimData(x,y,z)
        h = self.encoder(x)
        t = self.transformer.forward(x)
        cf = torch.cat((h, t), dim=1)

        # Go through all layers and apply h to them
        for l in range(self.num_layers):
            cf = self.backbone[l](cf)
            if l != self.num_layers - 1:
                
                cf = F.relu(cf, inplace=True)
                
                if l+1 in self.skips:
                    cf = torch.cat((cf, h, t), dim=1)

        colors = torch.sigmoid(cf[:,0])
        densities = F.relu(cf[:,1])

        return colors, densities