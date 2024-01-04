import os
import glob
import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

from rich.console import Console
from torch_ema import ExponentialMovingAverage

import packaging

from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import tifffile as tiff

def find_directory(dir_name):
    current_dir = os.getcwd()
    while current_dir != os.path.dirname(current_dir):  # stop at root directory
        check_dir = os.path.join(current_dir, dir_name)
        if os.path.isdir(check_dir):
            return check_dir
        current_dir = os.path.dirname(current_dir)
    return None

# Used in main.py to set seed for everything.
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def get_mgrid(x_dim, y_dim, z_dim):
    tensors = tuple([torch.linspace(-1, 1, steps=x_dim)] + [torch.linspace(-1, 1, steps=y_dim)] + [torch.linspace(-1, 1, steps=z_dim)])
    grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 3)
    grid = grid[grid[:, 2].argsort()]
    indices = np.lexsort((grid[:, 1], grid[:, 0], grid[:, 2]))
    grid = grid[indices]
    return grid

# def get_mgrid_single_slice(x_dim, y_dim, z_slice):
#     tensors = tuple([torch.linspace(-1, 1, steps=x_dim)] + [torch.linspace(-1, 1, steps=y_dim)] + [torch.tensor([z_slice])])
#     grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
#     grid = grid.reshape(-1, 3)
#     grid = (grid - 0) / grid.max()
#     return grid

def normalize(coordinates, x_dim, y_dim, z_dim):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    x_norm = 2*(x-(x_dim/2)) / x_dim
    y_norm = 2*(y-(y_dim/2)) / y_dim
    z_norm = 2*(z-(z_dim/2)) / z_dim

    norm_coordinates = torch.stack([x_norm, y_norm, z_norm], dim=1)
    return norm_coordinates

def get_slice_mgrid(x_dim, y_dim, z):

    tensors = tuple([torch.linspace(0, x_dim, x_dim)] + [torch.linspace(0, y_dim, y_dim)] + [torch.tensor(float(z))])
    grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 3)

    grid = normalize(grid, x_dim, y_dim, 1)

    grid = grid[grid[:, 0].argsort()]
    indices = np.lexsort((grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy()))
    grid = grid[indices]

    return grid

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

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 coarse_model,
                 fine_model,
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metirc
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 length=1,
                 num_cube_samples=64,
                 num_fine_samples=192
                 ):
        
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.length = length
        self.num_cube_samples = num_cube_samples
        self.num_fine_samples = num_fine_samples

        coarse_model.to(self.device)
        fine_model.to(self.device)
        self.coarse_model = coarse_model
        self.fine_model = fine_model        

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
            # Optimizer with both the coarse and fine model parameters
            self.optimizer = optim.Adam(list(self.coarse_model.parameters()) + list(self.fine_model.parameters()), lr=0.001, weight_decay=5e-4)
        else:
            # self.optimizer = optimizer(self.model)
            self.optimizer = optimizer(list(self.coarse_model.parameters()) + list(self.fine_model.parameters()))

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            # self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
            self.ema = ExponentialMovingAverage(list(self.coarse_model.parameters()) + list(self.fine_model.parameters()), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth.tar"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in list(coarse_model.parameters()) + list(fine_model.parameters()) if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()

    # TODO: May need to be modified
    def test(self, start, end, step, H, W, batch_size=100, base_image_path='pred'):
        self.log("[INFO] Generating test images ...")
        z_coords = torch.linspace(start, end, steps=step)
        for z_slice in z_coords:    
            coords = get_slice_mgrid(H, W, float(z_slice))
            self.test_image_partial(H, W, coords, self.device, batch_size=batch_size, imagepath=f'{base_image_path}_{z_slice:.2f}.png')

    def test_on_trained_slice(self, start_slice, end_slice, slice_index, base_img_path, imagepath='pred.png', resize_factor=1, save_img=True):
        self.log("[INFO] Generating test image slice ...")
        img_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=resize_factor, start_slice_index=slice_index, end_slice_index=slice_index)
        height, width = img_dataset.get_H_W()
        img_loader  = DataLoader(img_dataset, batch_size=height * width)

        img_pred = torch.empty(height * width, 1)
        img = torch.empty(height * width, 1)

        z_coord = torch.linspace(-1, 1, steps=end_slice-start_slice+1)[list(range(start_slice, end_slice+1)).index(slice_index)]

        i = 0
        for pixel, coords in img_loader:

            # Fix the z coordinates
            coords[:,2] = z_coord

            # Move input to GPU
            torch.cuda.empty_cache()
            coords = coords.to(self.device)

            # Get model prediction
            self.model.eval()
            pred = self.model(coords)

            # Store the partial prediction
            img_pred[i:i+len(coords)] = pred.cpu().detach()

            # Store the pixel values
            img[i:i+len(coords)] = pixel

            i = i + len(coords)

        # Calculate the loss
        mse = torch.nn.MSELoss()
        mse = mse(img_pred, img).item()
        psnr = 20 * np.log10(img.max() / np.sqrt(mse))

        # If should save predicted image and ground truth
        if save_img:
            model_img = img_pred.view(height,width).numpy()
            img = img.view(height,width).numpy()

            # Save prediction image
            plt.imsave(imagepath, model_img, cmap="gray")

            # Save ground truth image
            plt.imsave('gt.png', img, cmap="gray")


        print(f'The resulting MSE: {mse:.6f} and PSNR: {psnr}')

        return psnr


    def test_image_partial(self, height, width, coords, device, batch_size=100, imagepath='pred.png'):

        coord_batches = np.array_split(coords, (coords.shape[0] / batch_size))

        img_pred = torch.empty(coords.shape[0])

        i = 0
        for coord_batch in coord_batches:

            # Move input to GPU
            torch.cuda.empty_cache()
            coord_batch = coord_batch.to(device)
            
            # Get the coordinates and distances of the coarse samples for each center
            cube_samples = self.get_cube_samples(8, coord_batch, torch.tensor([self.length, self.length, self.length], device=self.device))
            coords = cube_samples[:, :, :3]
            distances = cube_samples[:, :, 3]

            # Predict the color and density of the coarse samples
            coords_flat = torch.reshape(coords, [-1, coords.shape[-1]])

            self.coarse_model.eval()
            colors_flat, densities_flat = self.coarse_model(coords_flat)

            # Reshape the flat output to the original input shape
            colors_c = torch.reshape(colors_flat, list(coords.shape[:-1]))
            densities_c = torch.reshape(densities_flat, list(coords.shape[:-1]))

            # Evaluate the coarse color for each center pixel
            course_colors = self.calculate_color(torch.cat((distances.unsqueeze(2), 
                                                    densities_c.unsqueeze(2), colors_c.unsqueeze(2)), dim=2))

            # Get fine samples for each center pixel
            fine_samples = self.get_cube_samples_hierarchical(8, distances, densities_c)

            # Predict the color and density of the fine samples
            fine_coords = torch.cat((coords, fine_samples[:, :,:3]), dim=1)
            fine_coords_flat = torch.reshape(fine_coords, [-1, fine_coords.shape[-1]])
            colors_flat_f, densities_flat_f = self.fine_model(fine_coords_flat)

            # Reshape the flat output to the original input shape
            colors_f = torch.reshape(colors_flat_f, list(fine_coords.shape[:-1]))
            densities_f = torch.reshape(densities_flat_f, list(fine_coords.shape[:-1]))

            # Concatenate the distances of the coarse and fine samples
            distances_f = torch.cat((distances, fine_samples[:, :, 3]), dim=1)

            inds = distances_f.argsort(dim=1)
            distances_f = distances_f.gather(1, inds)
            densities_f = densities_f.gather(1, inds)
            colors_f = colors_f.gather(1, inds)

            # Evaluate the fine color for each center pixel
            pred_colors = self.calculate_color(torch.cat((distances_f.unsqueeze(2), 
                                                densities_f.unsqueeze(2), colors_f.unsqueeze(2)), dim=2))

            img_pred[i:i+len(coord_batch)] = pred_colors.cpu().detach()
            i = i + len(coord_batch)

        # Calculate the loss
    #     mse = torch.nn.MSELoss()
    #     mse = mse(img_pred, img).item()
    #     psnr = 20 * log10(img.max() / sqrt(mse))

        # Show predicted image
        model_img = img_pred.view(height,width).numpy()

        f, ax = plt.subplots(1,1)
        ax.axis('off')
        ax.set_title('Neural Representation')
        ax.imshow(model_img, cmap="gray")
        f.show()

        # Save prediction image
        plt.imsave(imagepath, model_img, cmap="gray")

    #     print(f'The resulting MSE: {mse:.6f} and PSNR: {psnr}')

        return img_pred


    def log(self, *args, **kwargs):
        if not self.mute: 
            #print(*args)
            self.console.print(*args, **kwargs)
        if self.log_ptr: 
            print(*args, file=self.log_ptr)
            self.log_ptr.flush() # write immediately to file

    # Cube-based hierarchical sampling
    def get_cube_samples_hierarchical(self, n_samples, distances, densities):
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
        new_distances = self.sample_pdf(distances, densities, n_samples, det=False, pytest=False)

        # print("Max new distances: ", torch.max(new_distances))

        thetas = torch.rand((n_centers, n_samples), device=new_distances.device) * 2 * np.pi
        phis = torch.rand((n_centers, n_samples), device=new_distances.device) * np.pi
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
    def sample_pdf(self, bins, weights, N_samples, det=False, pytest=False):
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
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

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
        below = torch.max(torch.zeros_like(inds-1), torch.min(inds-1, (bins.shape[-1]-1) * torch.ones_like(inds)))
        above = torch.min((bins.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]

        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 1, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand((matched_shape[0], matched_shape[1], matched_shape[2]-1)), 1, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0]) * denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    # Adaptive Loss Function
    def adaptive_loss_fn(self, pixels, preds_coarse, preds_fine):
        """
        Calculates the adaptive loss defined as:
        L = sum( lambda * || pixels - preds_coarse ||_2 ** 2 + || pixels - preds_fine ||_2 ** 2)
        lambda = || pixels - preds_fine || ** 1/2
        """
        loss_fn = torch.nn.MSELoss()

        # Remove singleton dimension from pixels
        # pixels = pixels.squeeze(1)

        # MSE Loss between pixels and coarse predictions
        loss_coarse = loss_fn(preds_coarse, pixels)

        # MSE Loss between pixels and fine predictions
        loss_fine = loss_fn(preds_fine, pixels)

        # Adaptive Regularization Term 
        adapt_reg = torch.sqrt(loss_fine)

        # print(loss_coarse, loss_fine, adapt_reg)

        return adapt_reg * loss_coarse + loss_fine

    ### ------------------------------	
    # color, cube_samples
    def train_step(self, label, input):
        X = input
        y = label
        
        # Get the coordinates and distances of the coarse samples for each center
        coords = input[:, :, :3]
        distances = input[:, :, 3]

        # Predict the color and density of the coarse samples
        coords_flat = torch.reshape(coords, [-1, coords.shape[-1]])

        colors_flat, densities_flat = self.coarse_model(coords_flat)

        # Reshape the flat output to the original input shape
        colors_c = torch.reshape(colors_flat, list(coords.shape[:-1]))
        densities_c = torch.reshape(densities_flat, list(coords.shape[:-1]))

        # Evaluate the coarse color for each center pixel
        coarse_colors = self.calculate_color(torch.cat((distances.unsqueeze(2), 
                                                densities_c.unsqueeze(2), colors_c.unsqueeze(2)), dim=2))

        # Get fine samples for each center pixel
        fine_samples = self.get_cube_samples_hierarchical(self.num_fine_samples, distances, densities_c)

        # Predict the color and density of the fine samples
        fine_coords = torch.cat((coords, fine_samples[:, :,:3]), dim=1)
        fine_coords_flat = torch.reshape(fine_coords, [-1, fine_coords.shape[-1]])
        colors_flat_f, densities_flat_f = self.fine_model(fine_coords_flat)

        # Reshape the flat output to the original input shape
        colors_f = torch.reshape(colors_flat_f, list(fine_coords.shape[:-1]))
        densities_f = torch.reshape(densities_flat_f, list(fine_coords.shape[:-1]))

        # Concatenate the distances of the coarse and fine samples
        distances_f = torch.cat((distances, fine_samples[:, :, 3]), dim=1)

        inds = distances_f.argsort(dim=1)
        distances_f = distances_f.gather(1, inds)
        densities_f = densities_f.gather(1, inds)
        colors_f = colors_f.gather(1, inds)

        # Evaluate the fine color for each center pixel
        fine_colors = self.calculate_color(torch.cat((distances_f.unsqueeze(2), 
                                            densities_f.unsqueeze(2), colors_f.unsqueeze(2)), dim=2))

        # loss = self.criterion(pred_color, y.squeeze())
        loss = self.adaptive_loss_fn(y.squeeze(), coarse_colors, fine_colors)

        return fine_colors, y, loss

    def eval_step(self, label, input):
        return self.train_step(label, input)

    ### ------------------------------
    
    def train(self, train_loader, valid_loader, max_epochs):

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # For future, create lenghty list of X points over all data files for each Epoch.
            self.train_one_epoch(train_loader)

            if self.workspace is not None:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

    def evaluate(self, loader):
        self.evaluate_one_epoch(loader)

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.coarse_model.train()
        self.fine_model.train()
        
        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            self.local_step += 1
            self.global_step += 1
            
            colors = data[0].to(self.device)
            input_coords = data[1].to(self.device)

            # cube sampling.
            cube_samples = self.get_cube_samples(self.num_cube_samples, input_coords, torch.tensor([self.length, self.length, self.length], device=self.device))

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(colors, cube_samples)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.report_metric_at_train:
                for metric in self.metrics:
                    metric.update(preds, truths)
            if self.scheduler_update_every_step:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
            pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()
        if self.report_metric_at_train:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")


    def evaluate_one_epoch(self, loader):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        total_loss = 0
        for metric in self.metrics:
            metric.clear()

        self.coarse_model.eval()
        self.fine_model.eval()

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0
            for data in loader:    
                self.local_step += 1
                
                colors = data[0].to(self.device)
                input_coords = data[1].to(self.device)

                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()

                # cube sampling.
                cube_samples = self.get_cube_samples(self.num_cube_samples, input_coords, torch.tensor([self.length, self.length, self.length], device=self.device))
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.eval_step(colors, cube_samples)

                if self.ema is not None:
                    self.ema.restore()

                loss_val = loss.item()
                total_loss += loss_val

                for metric in self.metrics:
                    metric.update(preds, truths)

                pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        pbar.close()
        if not self.use_loss_as_metric and len(self.metrics) > 0:
            result = self.metrics[0].measure()
            self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
        else:
            self.stats["results"].append(average_loss) # if no metric, choose best by min loss

        for metric in self.metrics:
            self.log(metric.report(), style="blue")
            metric.clear()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    # Isotropic volumetric sampling
    def calculate_color(self, samples):
        n_centers, n_samples, _ = samples.shape

        # Calculate the differences between consecutive distances
        distance_diffs = samples[:, 1:, 0] - samples[:, :-1, 0]

        # Calculate the numerators
        numerators =  (samples[:, :-1, 0]**2) * (1 - torch.exp(-samples[:, :-1, 1] * distance_diffs))

        # Calculate the denominators
        denominators = torch.cumsum((samples[:, :-1, 0]**2) * samples[:, :-1, 1] * distance_diffs, dim=1)
        denominators = denominators * 4 * np.pi
        denominators = torch.exp(denominators)

        # Calculate the colors
        values = numerators / denominators * samples[:, :-1, 2]
        colors = torch.sum(values, dim=1)
        colors = colors * 4 * np.pi

        return colors

    def get_cube_samples(self, n_samples, centers, dimensions):
        samples = []
        for center in centers:
            center_samples = torch.rand((n_samples, 3)) - 0.5
            center_samples = center_samples.to(self.device)
            center_samples = center_samples * dimensions + center

            distances = torch.norm(center_samples - center, dim=-1, keepdim=True)
            center_samples = torch.cat([center_samples, distances], dim=-1)
            samples.append(center_samples)
        samples = torch.stack(samples, dim=0)
        for i, _ in enumerate(samples):
            samples[i] = samples[i][samples[i][:,3].argsort()]
        return samples

    def save_checkpoint(self, full=False, best=False):

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            # state['model'] = self.model.state_dict()
            state['coarse_model'] = self.coarse_model.state_dict()
            state['fine_model'] = self.fine_model.state_dict()

            file_path = f"{self.ckpt_path}/{self.name}_ep{self.epoch:04d}.pth.tar"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = self.stats["checkpoints"].pop(0)
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    # state['model'] = self.model.state_dict()
                    state['coarse_model'] = self.coarse_model.state_dict()
                    state['fine_model'] = self.fine_model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth.tar'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            # self.model.load_state_dict(checkpoint_dict)
            self.coarse_model.load_state_dict(checkpoint_dict['coarse_model'])
            self.fine_model.load_state_dict(checkpoint_dict['fine_model'])
            self.log("[INFO] loaded model.")
            return

        # missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        missing_keys, unexpected_keys = self.coarse_model.load_state_dict(checkpoint_dict['coarse_model'], strict=False)
        missing_keys, unexpected_keys = self.fine_model.load_state_dict(checkpoint_dict['fine_model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")            

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer, use default.")
        
        # strange bug: keyerror 'lr_lambdas'
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 5e-4
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler, use default.")

        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])  