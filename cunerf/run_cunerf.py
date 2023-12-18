import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from load_tiff import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def render_view(model, H, W, translation, rotation, batch_size, savedir):

    print(f'Rendering new view with {translation} and {rotation}')

    # Generate new coordinates for the current view
    coords = get_view_mgrid(H, W, translation, rotation)
    
    rgbs = []
    # Predict RGB for each coordinate
    for i in range(0, coords.shape[0], batch_size):
        batch_coords = coords[i:i+batch_size]
        batch_coords = torch.Tensor(batch_coords).to(device)
        print(batch_coords.shape)
        print(batch_coords)
        colors, densities = model(batch_coords)

        print(colors.shape, densities.shape)

        rgbs.append(colors.cpu().detach().numpy())
    
    # Save the predicted RGBs as a 16-bit grayscale image
    if savedir is not None:
        rgbs = np.concatenate(rgbs, axis=0)
        rgbs = np.reshape(rgbs, (H, W))
        rgbs = np.clip(rgbs, 0, 1)
        rgbs = (rgbs * 65535).astype(np.uint16)
        filename = os.path.join(savedir, '{:03d}.png'.format(i))

        rgbs = np.array(rgbs, dtype=np.uint16)
        print(rgbs.shape)

        plt.imsave(filename, rgbs, cmap="gray")

    print('Rendering new view finished.')


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default='test', 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')
    parser.add_argument("--base_img_name", type=str, default='./dataset/pp_174_tumor_Nr56_x4_StitchPag_stitch_2563x4381x2162', help="base image name")
    parser.add_argument("--start_index", type=str, default=11390, help="start index")
    parser.add_argument("--end_index", type=str, default=11399, help="end index")
    parser.add_argument("--ft_path", type=str, default=None)

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--fp16", action='store_true',)
    parser.add_argument("--print_interval", type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--save_interval", type=int, default=10000,
                        help='frequency of metric saving')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')

    return parser

def train_step(model_coarse, model_fine, label, input, length, n_samples_c, n_samples_f):
    X = input # [N, 3]
    y = label
    
    # Sample for each point in X in a cube around it
    X_coarse_samples = get_cube_samples(n_samples_c, X, length)

    # Get the coordinates and distances of the coarse samples for each center
    coords = X_coarse_samples[:, :, :3]
    distances = X_coarse_samples[:, :, 3]

    # Predict the color and density of the coarse samples
    coords_flat = torch.reshape(coords, [-1, coords.shape[-1]])
    colors_flat_c, densities_flat_c = model_coarse(coords_flat)

    # Reshape the flat output to the original input shape
    colors_c = torch.reshape(colors_flat_c, list(coords.shape[:-1]))
    densities_c = torch.reshape(densities_flat_c, list(coords.shape[:-1]))

    # Evaluate the coarse color for each center pixel
    coarse_center_color = calculate_color(torch.cat((distances.unsqueeze(2), 
                                          densities_c.unsqueeze(2), colors_c.unsqueeze(2)), dim=2))

    # Get fine samples for each center pixel
    fine_samples = get_cube_samples_hierarchical(n_samples_f, distances, densities_c)

    # Predict the color and density of the fine samples
    fine_coords = torch.cat((coords, fine_samples[:, :,:3]), dim=1)
    fine_coords_flat = torch.reshape(fine_coords, [-1, fine_coords.shape[-1]])
    colors_flat_f, densities_flat_f = model_fine(fine_coords_flat)

    # Reshape the flat output to the original input shape
    colors_f = torch.reshape(colors_flat_f, list(fine_coords.shape[:-1]))
    densities_f = torch.reshape(densities_flat_f, list(fine_coords.shape[:-1]))

    # Concatenate the distances of the coarse and fine samples
    distances_f = torch.cat((distances, fine_samples[:, :, 3]), dim=1)

    # TODO: May need to sort the samples by distance

    # Evaluate the fine color for each center pixel
    fine_center_color = calculate_color(torch.cat((distances_f.unsqueeze(2), 
                                        densities_f.unsqueeze(2), colors_f.unsqueeze(2)), dim=2))

    loss = adaptive_loss_fn(y, coarse_center_color, fine_center_color)

    return fine_center_color, y, loss
    
def load_checkpoint(args, basedir, expname, optimizer, model_coarse, model_fine):
     # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    start = 0
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model_coarse.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
    return start, optimizer, model_coarse, model_fine

def cuNeRF_train():
    parser = config_parser()
    args = parser.parse_args()

    base_img_name = args.base_img_name
    start_index = args.start_index
    end_index = args.end_index

    # Load data and create models
    colors, coords, H, W = load_tiff_images(start_index, end_index, base_img_name, resize_factor=10)
    H, W = int(H), int(W)

    model_coarse = CuNeRF(D=9, W=256, W_last=128, input_ch=3, output_ch=2, skips=[3, 7]) 
    model_fine = CuNeRF(D=9, W=256, W_last=128, input_ch=3, output_ch=2, skips=[3, 7])

    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    print('basedir', basedir)
    print('expname', expname)
    start = 0
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    start, optimizer, model_coarse, model_fine = load_checkpoint(args, basedir, expname, optimizer, model_coarse, model_fine)

    global_step = start

    # Short circuit if only rendering out from trained model
    if args.render_only:
        with torch.no_grad():
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}.png'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)

            batch_size = 10000
            render_view(model_fine, H, W, [0, 0, 0], [0, 0, 0], batch_size, testsavedir)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    # Create dataset and dataloader for train and validation set
    num_samples = 10
    train_dataset = MicroCTVolume(colors, coords, H, W)
    train_loader = DataLoader(train_dataset, batch_size=num_samples, shuffle=True, generator=torch.Generator(device='cuda'))

    valid_dataset = MicroCTVolume(colors, coords, H, W)
    valid_loader  = DataLoader(valid_dataset, batch_size=num_samples, generator=torch.Generator(device='cuda'))

    N_epocs = 10
    print('Begin')

    model_coarse.train()
    model_fine.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    global_step = 0

    min_loss = torch.inf
    for epoch in range(N_epocs):

        print(f'Epoch {epoch}...')
        
        local_step = 0
        
        for batch in train_loader:
            
            colors = batch[0].to(device)
            coords = batch[1].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):
                preds, truths, loss = train_step(model_coarse, model_fine, colors, coords, 1, 64, 192)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                
            if global_step % args.print_interval == 0:
                tqdm.write(f"[TRAIN] Step: {global_step} Loss: {loss}")
            
            # Save checkpoint 
            if global_step % args.save_interval==0:
                if loss <= min_loss:
                    min_loss = loss
                    path = os.path.join(basedir, expname, '{:06d}.tar'.format(global_step))
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': model_coarse.state_dict(),
                        'network_fine_state_dict': model_fine.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    print('Saved checkpoints at', path)

            # Save the latest ckpt every 500 steps
            if global_step % 500 == 0 and global_step > 0:
                path = os.path.join(basedir, expname, '{:06d}_latest.tar'.format(global_step))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': model_coarse.state_dict(),
                    'network_fine_state_dict': model_fine.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            global_step += 1
            local_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    cuNeRF_train()
