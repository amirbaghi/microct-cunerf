import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# Dataset 
class MicroCTVolume(Dataset):
    def __init__(self, colors, coords, H, W):
        self.colors = colors
        self.coords = coords
        self.H = H
        self.W = W
        
    def get_H_W(self):
        return self.H, self.W
    
    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if idx > self.coords.shape[0]:
            raise IndexError("Index out of range")
        
        coords = self.coords[idx]
        
        colors = self.colors.view(-1,1)[idx]
        
        return colors, coords

# CuNeRF Model
class CuNeRF(nn.Module):
    def __init__(self, D=9, W=256, W_last=128, input_ch=3, output_ch=2, skips=[4, 7], freq=10):
        super(CuNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.freq = freq
        self.freq_range = torch.arange(self.freq, device='cuda')
            
        self.linears = nn.ModuleList(
            # [nn.Linear(input_ch * self.freq * 2, W)] + [nn.Linear(W, W if i != D-2 else W_last) if i not in self.skips else nn.Linear(W + (input_ch * self.freq * 2), W if i != D-2 else W_last) for i in range(D-1)])
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W if i != D-2 else W_last) if i not in self.skips else nn.Linear(W + input_ch, W if i != D-2 else W_last) for i in range(D-1)])

        self.output_linear = nn.Linear(W_last, output_ch)

    # Positional encoding
    def positional_encoding(self, x):
        # x = x.unsqueeze(-1) # TODO: may need to remove.
        # Create a tensor of powers of 2
        scales = 2.0 ** self.freq_range
        # Compute sin and cos features
        features = torch.cat([torch.sin(x * np.pi * scale) for scale in scales] + [torch.cos(x * np.pi * scale) for scale in scales], dim=-1)
        return features

    def forward(self, x):
        # TODO: We can encode the iputs in hash-grids

        # h = self.positional_encoding(x)
        h = x

        for i, l in enumerate(self.linears):

            h = self.linears[i](h)
            h = F.relu(h)

            if i in self.skips:
                # encoded_inps = self.positional_encoding(x)
                encoded_inps = x
                h = torch.cat([encoded_inps, h], -1)

        outputs = self.output_linear(h)

        colors = torch.sigmoid(outputs[:,0])
        densities = F.relu(outputs[:,1])

        return colors, densities

# Adaptive Loss Function
def adaptive_loss_fn(pixels, preds_coarse, preds_fine):
    """
    Calculates the adaptive loss defined as:
    L = sum( lambda * || pixels - preds_coarse ||_2 ** 2 + || pixels - preds_fine ||_2 ** 2)
    lambda = || pixels - preds_fine || ** 1/2
    """
    loss_fn = torch.nn.MSELoss()

    # Remove singleton dimension from pixels
    pixels = pixels.squeeze(1)

    print("Pixels: ", pixels)
    print("Preds_coarse:", preds_coarse)
    print("Preds_fine:", preds_fine, end="\n\n")

    # MSE Loss between pixels and coarse predictions
    loss_coarse = loss_fn(preds_coarse, pixels)

    print(loss_coarse)

    # MSE Loss between pixels and fine predictions
    loss_fine = loss_fn(preds_fine, pixels)

    print(loss_fine)

    # Adaptive Regularization Term
    # || pixels - preds_fine || ** 1/2
    # adapt_reg = torch.sqrt(torch.mean((pixels - preds_fine) ** 2))
    

    # return adapt_reg * loss_coarse + loss_fine
    return loss_coarse + loss_fine

# Cube-sampling
def get_cube_samples(n_samples, centers, length):
    """
    Get samples on a 3D cube for each center and return the coordinates and their distances to the centers
    The returned coordinates are sorted by their distances to the centers
    return: (centers, n_samples, 4) => (x, y, z, distance)
    """

    samples = []
    for center in centers:
        center_samples = torch.rand((n_samples, 3)) - 0.5
        center_samples = center_samples * length + center
        distances = torch.norm(center_samples - center, dim=-1, keepdim=True)
        center_samples = torch.cat([center_samples, distances], dim=-1)
        samples.append(center_samples)
    samples = torch.stack(samples, dim=0)
    for i, _ in enumerate(samples):
        samples[i] = samples[i][samples[i][:,3].argsort()]
    return samples

# Isotropic volumetric sampling
def calculate_color(samples):
    """
    Calculate the color of center points from the samples
    Formula: 4*pi*sum(distance_i^2 * (1 - exp(-density_i(r_i_+1 - r_i)))/exp(4*pi*sum_1_i(distance_i^2*density_j*(distance_j_+1 - distance_j))) * color_i)
    samples: (n_centers, n_samples, 3): (distance, density, color)
    return: (n_centers, 3)
    """

    # Print the nan values in the samples
    # print("Samples: ", samples)
    # print("Samples Nan values: ", torch.isnan(samples).any())

    n_centers, n_samples, _ = samples.shape

    # Calculate the differences between consecutive distances
    distance_diffs = samples[:, 1:, 0] - samples[:, :-1, 0]

    # Calculate the numerators
    numerators = samples[:, :-1, 0]**2 * (1 - torch.exp(-samples[:, :-1, 1] * distance_diffs))

    # Calculate the denominators
    denominators = torch.cumsum(samples[:, :-1, 0]**2 * samples[:, :-1, 1] * distance_diffs, dim=1)
    denominators *= 4 * np.pi
    denominators = torch.exp(denominators)

    # Calculate the colors
    colors = torch.sum(numerators / denominators * samples[:, :-1, 2], dim=1)
    colors *= 4 * np.pi

    return colors

def evaluate_point(model, samples, device):
    # Get the color of a point using the samples around it in a 3D cube
    # samples: (n_samples, 4) => (x, y, z, distance)
    # return: (3,)

    # Get density and color for each sample from the cunerf model
    samples = samples.unsqueeze(0)
    samples = samples.to(device)
    colors, densities = model(samples)

    # Concatenate each sample's distance, density and color
    samples = torch.cat([samples[:,3], densities, colors], dim=-1)

    # Calculate the color of the point
    color = calculate_color(samples)
    
    return color

# Cube-based hierarchical sampling
def get_cube_samples_hierarchical(n_samples, distances, densities):
    """
    Generate new coordinates using the new distances
    Using the formula x = (distance * sin(phi) * cos(theta), distance * sin(phi) * sin(theta), distance * cos(phi))
    theta and phi are sampled from a uniform distribution
    theta: [0, 2*pi]
    phi: [0, pi]
    distances: (n_centers, n_coarse_samples)
    densities: (n_centers, n_coarse_samples)
    return: (n_centers, n_samples, 4) => (x, y, z, distance)
    """

    n_centers = distances.shape[0]

    # Get new distances using ITS on the distances and densities of the course samples
    new_distances = sample_pdf(distances, densities, n_samples, det=False, pytest=False)

    thetas = torch.rand((n_centers, n_samples)) * 2 * np.pi
    phis = torch.rand((n_centers, n_samples)) * np.pi
    x = new_distances * torch.sin(phis) * torch.cos(thetas)
    y = new_distances * torch.sin(phis) * torch.sin(thetas)
    z = new_distances * torch.cos(phis)
    new_samples = torch.stack([x, y, z], dim=-1)

    # Add distances to the result tensor for each element
    new_samples = torch.cat([new_samples, new_distances.unsqueeze(-1)], dim=-1)

    # Sort the resulting tensor by the distance
    # Get the indices that would sort the distances
    sorted_indices = new_samples[:, :, 3].argsort(dim=1)

    # Use these indices to sort new_samples
    new_samples = new_samples.gather(1, sorted_indices.unsqueeze(-1).expand(-1, -1, new_samples.size(-1)))

    return new_samples

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """
    Sample from discretized pdf
    bins: (n_centers, n_coarse_samples)
    weights: (n_centers, n_coarse_samples)
    return: (n_centers, n_samples)
    """

    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((bins.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand((matched_shape[0], matched_shape[1], matched_shape[2]-1)), 1, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
