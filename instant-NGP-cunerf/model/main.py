import torch
import optuna
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

from ngp_network_RHINO import INGPNetworkRHINO
from ngp_network import INGPNetwork
from utils import *
from load_tiff import load_tiff_images
from skimage.metrics import structural_similarity
from torchmetrics.functional import peak_signal_noise_ratio

def train_model(trial, start_slice, end_slice, base_img_path, lr, fp16, workspace, resize_factor, maximum_parameters=8000000):

    # Load the data
    colors, coords, H, W = load_tiff_images(start_slice, end_slice, base_img_name, resize_factor=resize_factor)
    H, W = int(H), int(W)

    # Suggest values for the hyperparameters
    num_layers_c = trial.suggest_int('num_layers_c', 5, 9)
    num_layers_f = trial.suggest_int('num_layers_f', 5, 9)
    
    hidden_dim_c = trial.suggest_categorical('hidden_dim_c', [128, 256, 512])
    hidden_dim_f = trial.suggest_categorical('hidden_dim_f', [128, 256, 512])

    num_levels_c = trial.suggest_categorical('num_levels_c', [2, 4, 8, 16, 32, 64])
    num_levels_f = trial.suggest_categorical('num_levels_f', [2, 4, 8, 16, 32, 64])

    level_dim_c = trial.suggest_categorical('level_dim_c', [2, 4, 8])
    level_dim_f = trial.suggest_categorical('level_dim_f', [2, 4, 8])
    
    base_resolution_c = trial.suggest_categorical('base_resolution_c', [8, 16, 32])
    base_resolution_f = trial.suggest_categorical('base_resolution_f', [8, 16, 32])

    log2_hashmap_size_c = trial.suggest_int('log2_hashmap_size_c', 10, 20)
    log2_hashmap_size_f = trial.suggest_int('log2_hashmap_size_f', 10, 21)

    desired_resolution_f = trial.suggest_categorical('desired_resolution_f', [H / 2, H, 2 * H])

    align_corners = trial.suggest_categorical('align_corners', [True, False])

    freq_c = trial.suggest_categorical('freq_c', [10, 20, 40, 50, 70, 80])
    freq_f = trial.suggest_categorical('freq_f', [10, 20, 40, 50, 70, 80])

    transformer_layers_c = trial.suggest_categorical('transformer_layers_c', [1])
    transformer_layers_f = trial.suggest_categorical('transformer_layers_f', [1])

    transformer_neurons_c = trial.suggest_categorical('transformer_neurons_c', [32, 64, 128, 256, 512])
    transformer_neurons_f = trial.suggest_categorical('transformer_neurons_f', [32, 64, 128, 256, 512])

    cube_xy_length = trial.suggest_categorical('cube_xy_length', [0.00001, 0.0001, 0.0005, 1. / H, 2. / H, 4. / H])
    cube_z_length = trial.suggest_categorical('cube_z_length', [1. / (end_slice - start_slice + 1), 2. / (end_slice - start_slice + 1), 4. / (end_slice - start_slice + 1)])


    # Set the length of the cube
    cube_lengths = torch.tensor([cube_xy_length, cube_xy_length, cube_z_length], device='cuda')

    # Initialize the models
    coarse_model = INGPNetworkRHINO(num_layers=num_layers_c, hidden_dim=hidden_dim_c, skips=[4, 7], input_dim=3, num_levels=num_levels_c,
                    level_dim=level_dim_c, base_resolution=base_resolution_c, log2_hashmap_size=log2_hashmap_size_c, desired_resolution=H, 
                    align_corners=align_corners, freq=freq_c, transformer_num_layers=transformer_layers_c, transformer_hidden_dim=transformer_neurons_c)
    fine_model = INGPNetworkRHINO(num_layers=num_layers_f, hidden_dim=hidden_dim_f, skips=[4, 7], input_dim=3, num_levels=num_levels_f,
            level_dim=level_dim_f, base_resolution=base_resolution_f, log2_hashmap_size=log2_hashmap_size_f, desired_resolution=desired_resolution_f, 
            align_corners=align_corners, freq=freq_f, transformer_num_layers=transformer_layers_f, transformer_hidden_dim=transformer_neurons_f)

    # Check to see if the number of all parameters is less than the maximum
    num_params = sum([p.numel() for p in list(coarse_model.parameters()) + list(fine_model.parameters()) if p.requires_grad])
    if num_params > maximum_parameters:
        return 0

    # Set the number of samples inside the cube
    num_coarse_samples = 64
    num_fine_samples = 192

    # Create dataset and dataloader for train and validation set
    num_samples = 5000
    train_dataset = MicroCTVolume(colors, coords, H, W)
    train_loader = DataLoader(train_dataset, batch_size=num_samples, shuffle=True, generator=torch.Generator(device='cpu'))

    valid_dataset = MicroCTVolume(colors[0], coords.view(colors.shape[0], H, W, 3)[0].view(-1, 3), H, W)
    valid_loader  = DataLoader(valid_dataset, batch_size=10000, generator=torch.Generator(device='cpu'))

    optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': coarse_model.encoder.parameters()},
            {'name': 'transformer', 'params': coarse_model.transformer.parameters(), 'weight_decay': 1e-6},
            {'name': 'net', 'params': coarse_model.backbone.parameters(), 'weight_decay': 1e-6},
            {'name': 'encoding', 'params': fine_model.encoder.parameters()},
            {'name': 'transformer', 'params': fine_model.transformer.parameters(), 'weight_decay': 1e-6},
            {'name': 'net', 'params': fine_model.backbone.parameters(), 'weight_decay': 1e-6},
    ], lr=lr, betas=(0.9, 0.99), eps=1e-15)

    # Create scheduler to reduce the step size after N many epochs.
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize the trainer.
    trainer = Trainer('ngp', coarse_model, fine_model, workspace=workspace, optimizer=optimizer, ema_decay=0.95, fp16=fp16, lr_scheduler=scheduler, use_checkpoint='scratch',
                        eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)
    
    # Train and evaluate
    try:
        # Train the network
        trainer.train(train_loader, valid_loader, 5, H, W)

        # Test out the model
        avg_psnr, avg_ssim, avg_lpips = trainer.create_ground_truths(H, W, start_slice, end_slice, colors, coords)

        return avg_psnr

    except:

        return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rhino', action='store_true', help="RHINO mode")
    parser.add_argument('--tune', action='store_true', help="tune hyperparameters")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_on_slice', action='store_true', help="test on slice")
    parser.add_argument('--test_slice', type=int, default=0, help="test on slice index")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--batch_size', type=int, default=2000000, help="batch size (i.e. number of random samples)")
    parser.add_argument('--start_slice', type=int, default=11390, help="start slice")
    parser.add_argument('--end_slice', type=int, default=11399, help="end slice")
    parser.add_argument('--base_img_name', type=str, default='dataset/pp_174_tumor_Nr56_x4_StitchPag_stitch_2563x4381x2162', help="base image name")
    parser.add_argument('--render_new_view', action='store_true', help="render new view")
    parser.add_argument('--render_full_rotation', action='store_true', help="render full rotation")
    parser.add_argument('--translation', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="translation")
    parser.add_argument('--rotation_angle', type=float, default=0.0, help="rotation angles")
    parser.add_argument('--rotation_axis', type=float, nargs='+', default=[1.0, 0.0, 0.0], help="rotation axis")
    parser.add_argument('--reconstruct', action='store_true', help="reconstruct")

    opt = parser.parse_args()
    seed_everything(opt.seed)

    start_slice = opt.start_slice
    end_slice = opt.end_slice
    base_img_name = opt.base_img_name

    colors, coords, H, W = load_tiff_images(start_slice, end_slice, base_img_name, resize_factor=4)
    H, W = int(H), int(W)
    cube_lengths = torch.tensor([0.0001, 0.0001, 2. / (end_slice - start_slice + 1)], device='cuda')

    # Set the number of samples inside the cube
    num_coarse_samples = 64
    num_fine_samples = 192

    # Add the paramters for the network.
    if opt.rhino:
        # Use this! cunerf-rhino-resize4
        coarse_model = INGPNetworkRHINO(num_layers=9, hidden_dim=128, skips=[4, 7], input_dim=3, num_levels=8,
                        level_dim=2, base_resolution=16, log2_hashmap_size=10, desired_resolution=H, 
                        align_corners=False, freq=20, transformer_num_layers=1, transformer_hidden_dim=32)
        fine_model = INGPNetworkRHINO(num_layers=9, hidden_dim=256, skips=[4, 7], input_dim=3, num_levels=16,
                level_dim=4, base_resolution=16, log2_hashmap_size=16, desired_resolution=2*H, 
                align_corners=False, freq=50, transformer_num_layers=1, transformer_hidden_dim=128)

       # Use this! cunerf-rhino-2-r4
        # coarse_model = INGPNetworkRHINO(num_layers=5, hidden_dim=128, skips=[3], input_dim=3, num_levels=16,
        #                 level_dim=2, base_resolution=16, log2_hashmap_size=12, desired_resolution=H, 
        #                 align_corners=False, freq=20, transformer_num_layers=1, transformer_hidden_dim=32)
        # fine_model = INGPNetworkRHINO(num_layers=5, hidden_dim=256, skips=[3], input_dim=3, num_levels=16,
        #         level_dim=2, base_resolution=16, log2_hashmap_size=18, desired_resolution=H, 
        #         align_corners=False, freq=50, transformer_num_layers=1, transformer_hidden_dim=128)
    else:
        model = INGPNetwork(num_layers=5, hidden_dim=512, input_dim=3, num_levels=17, 
                        level_dim=4, base_resolution=16, log2_hashmap_size=21, desired_resolution=261, 
                        align_corners=False)


    dataset_dir = find_directory('dataset')
    if dataset_dir is None:
        print("Dataset directory not found. Please download the dataset as described in README.md.")
        exit(1)
    
    base_img_path = os.path.join(dataset_dir, base_img_name)

    # Reconstruct the given slices from start_slice to end_slice and report the PSNR, SSIM and LPIPS
    if opt.reconstruct:
        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)
        H, W = int(H), int(W)
        trainer.create_ground_truths(H, W, opt.start_slice, opt.end_slice, colors, coords)

    # Tune the hyperparameters
    elif opt.tune:
        study = optuna.create_study(direction='maximize', study_name='ngp_study', storage='sqlite:////cephyr/users/amirmaso/Alvis/microct-neural-repr/ngp_study.db', load_if_exists=True)
        study.optimize(lambda trial: train_model(trial, start_slice, end_slice, base_img_path, opt.lr, opt.fp16, opt.workspace, resize_factor=4), n_trials=100)
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

    # Test the model by reconstructing the slices from start_slice to end_slice with in-between slices (i.e. generalization)
    elif opt.test:
        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)
        H, W = int(H), int(W)
        trainer.test(0, (end_slice-start_slice+1), 1, H, W, colors, batch_size=5000)

    # Render new views of the volume given the translation, rotation angle and rotation axis
    elif opt.render_new_view:
        print(f'Rendering new view...')

        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)

        # Renders a full 360 rotation of the volume around the given rotation axis
        if opt.render_full_rotation:

            for x_rot in range(0, 360, 10):
                translation = opt.translation
                rotation_angle = x_rot
                rotation_axis = opt.rotation_axis
                view_coords = get_view_mgrid(H, W, translation, rotation_angle, rotation_axis)

                prediction = trainer.test_image_partial(H, W, view_coords, 'cuda', batch_size=5000, imagepath=f'new_view_{x_rot}.png')

        else:
            translation = opt.translation
            rotation_angle = opt.rotation_angle
            rotation_axis = opt.rotation_axis
            view_coords = get_view_mgrid(H, W, translation, rotation_angle, rotation_axis)

            prediction = trainer.test_image_partial(H, W, view_coords, 'cuda', batch_size=5000, imagepath=f'new_view.png')

    else:

        # Create dataset and dataloader for train and validation set
        num_samples = 5000
        train_dataset = MicroCTVolume(colors, coords, H, W)
        train_loader = DataLoader(train_dataset, batch_size=num_samples, shuffle=True, generator=torch.Generator(device='cpu'))

        valid_dataset = MicroCTVolume(colors[0], coords.view(colors.shape[0], H, W, 3)[0].view(-1, 3), H, W)
        valid_loader  = DataLoader(valid_dataset, batch_size=10000, generator=torch.Generator(device='cpu'))

        # Load the optimiser (Adam) with the encoding parameters (resulting hashed values when encoding) and the layers themselves.
        if opt.rhino:
            optimizer = lambda model: torch.optim.Adam([
                {'name': 'encoding', 'params': coarse_model.encoder.parameters()},
                {'name': 'transformer', 'params': coarse_model.transformer.parameters(), 'weight_decay': 1e-6},
                {'name': 'net', 'params': coarse_model.backbone.parameters(), 'weight_decay': 1e-6},
                {'name': 'encoding', 'params': fine_model.encoder.parameters()},
                {'name': 'transformer', 'params': fine_model.transformer.parameters(), 'weight_decay': 1e-6},
                {'name': 'net', 'params': fine_model.backbone.parameters(), 'weight_decay': 1e-6},
            ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        else:
            optimizer = lambda model: torch.optim.Adam([
                {'name': 'encoding', 'params': model.encoder.parameters()},
                {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
            ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        # Create scheduler to reduce the step size after N many epochs.
        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Initialize the trainer.
        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)
        
        # Train the network
        trainer.train(train_loader, valid_loader, 15, H, W)

        # Evaluate the training
        H,W = train_dataset.get_H_W()
        trainer.test(0, (end_slice-start_slice+1), 1, H, W, colors, batch_size=5000)
