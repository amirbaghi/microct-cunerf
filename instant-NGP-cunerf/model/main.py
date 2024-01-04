import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
import optuna

from ngp_network_RHINO import INGPNetworkRHINO
from ngp_network import INGPNetwork
from utils import *
from load_tiff import *

def train_model(trial, start_slice, end_slice, base_img_path, lr, fp16, workspace, rhino):
    # Suggest values for the hyperparameters
    num_layers = trial.suggest_int('num_layers', 3, 4)
    hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
    num_levels = trial.suggest_categorical('num_levels', [2, 4, 8, 16, 32, 64])
    base_resolution = trial.suggest_categorical('base_resolution', [8, 16, 32])
    log2_hashmap_size = trial.suggest_categorical('log2_hashmap_size', [19, 20, 21])
    desired_resolution = trial.suggest_categorical('desired_resolution', [256, 1024, 2048, 2088, 4096])
    align_corners = trial.suggest_categorical('align_corners', [True, False])
    freq = trial.suggest_categorical('freq', [10, 20, 40, 50, 70, 80])
    transformer_neurons = trial.suggest_categorical('transformer_neurons', [32, 64, 128, 256, 512])
    transformer_layers = trial.suggest_categorical('transformer_layers', [1])
    
    # Add the paramters for the network.
    if rhino:
        model = INGPNetworkRHINO(num_layers=num_layers, hidden_dim=hidden_dim, input_dim=3, num_levels=num_levels, 
                        level_dim=2, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, 
                        align_corners=align_corners, freq=freq, transformer_num_layers=transformer_layers, transformer_hidden_dim=transformer_neurons)
    else:
        model = INGPNetwork(num_layers=num_layers, hidden_dim=hidden_dim, input_dim=3, num_levels=num_levels, 
                        level_dim=2, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, 
                        align_corners=align_corners)

    # Create the training data and load it.
    train_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=7, start_slice_index=start_slice, end_slice_index=end_slice)

    H, W = train_dataset.get_H_W()
    random_samples = H * W  # TODO: May have to cap this to the maximum number of samples that can be loaded into GPU memory.
    train_loader = DataLoader(train_dataset, batch_size=random_samples, shuffle=True)

    # Create the validdation data and load it
    valid_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=7, start_slice_index=start_slice, end_slice_index=end_slice)
    valid_loader  = DataLoader(valid_dataset, batch_size=random_samples)

    # Use MSELoss as criterion (loss/error function), TODO: Might need change.
    criterion = torch.nn.MSELoss()

    # Load the optimiser (Adam) with the encoding parameters (resulting hashed values when encoding) and the layers themselves.
    if rhino:
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'transformer', 'params': model.transformer.parameters(), 'weight_decay': 1e-6},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
    else:
        optimizer = lambda model: torch.optim.Adam([
            {'name': 'encoding', 'params': model.encoder.parameters()},
            {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
        ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

    # Create scheduler to reduce the step size after N many epochs.
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Initialize the trainer.
    trainer = Trainer('ngp', model, workspace=workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=fp16, lr_scheduler=scheduler, use_checkpoint='scratch', eval_interval=1)

    # Train the network
    trainer.train(train_loader, valid_loader, 10)

    # Evaluate the training
    try:
        total_psnr = 0
        # for i in range(start_slice, end_slice + 1):
        for i in range(start_slice, start_slice + 1):
            psnr = trainer.test_on_trained_slice(start_slice, end_slice, i, base_img_path, imagepath=f'./tmp.png', resize_factor=1, save_img=False)
            total_psnr += psnr

        # return total_psnr / (end_slice - start_slice + 1)
        return total_psnr
    except:
        return 0
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rhino', action='store_true', help="RHINO mode")
    parser.add_argument('--tune', action='store_true', help="tune hyperparameters")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_on_slice', action='store_true', help="test on slice")
    parser.add_argument('--test_slice', type=int, default=11390, help="test on slice index")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--batch_size', type=int, default=2000000, help="batch size (i.e. number of random samples)")
    parser.add_argument('--start_slice', type=int, default=11390, help="start slice")
    parser.add_argument('--end_slice', type=int, default=11399, help="end slice")
    parser.add_argument('--base_img_name', type=str, default='dataset/pp_174_tumor_Nr56_x4_StitchPag_stitch_2563x4381x2162', help="base image name")

    opt = parser.parse_args()
    seed_everything(opt.seed)
    # Add the paramters for the network.
    if opt.rhino:
        model = INGPNetworkRHINO(num_layers=5, hidden_dim=512, input_dim=3, num_levels=17,
                        level_dim=2, base_resolution=16, log2_hashmap_size=21, desired_resolution=2088, 
                        align_corners=False)
    #     model = INGPNetworkRHINO(num_layers=4, hidden_dim=256, input_dim=3, num_levels=16, # num_levels=17
    #                     level_dim=2, base_resolution=8, log2_hashmap_size=22, desired_resolution=1024,  # desired_resolution=2048
    #                     align_corners=False, freq=20, transformer_num_layers=1, transformer_hidden_dim=64)
    else:
        model = INGPNetwork(num_layers=5, hidden_dim=512, input_dim=3, num_levels=17, 
                        level_dim=4, base_resolution=16, log2_hashmap_size=21, desired_resolution=261, 
                        align_corners=False)

    start_slice = opt.start_slice
    end_slice = opt.end_slice
    base_img_name = opt.base_img_name

    dataset_dir = find_directory('dataset')
    if dataset_dir is None:
        print("Dataset directory not found. Please download the dataset as described in README.md.")
        # Exit the program.
        exit(1)
    
    base_img_path = os.path.join(dataset_dir, base_img_name)

    if opt.test:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        # Evaluate the training
        # train_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=1, start_slice_index=start_slice, end_slice_index=end_slice)
        # H,W = train_dataset.get_H_W()
        colors, coords, H, W = load_tiff_images(start_slice, end_slice, base_img_name, resize_factor=4)
        H, W = int(H), int(W)
        trainer.test(-1, 1, (end_slice-start_slice+1), H, W, batch_size=1000)

    elif opt.test_on_slice:
        trainer = Trainer('ngp', model, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='best', eval_interval=1)
        # Evaluate the training
        trainer.test_on_trained_slice(start_slice, end_slice, opt.test_slice, base_img_path, resize_factor=1, imagepath=f'./pred-{opt.test_slice}.png')

    elif opt.tune:
        # Create the study object and optimize the hyperparameters.
        study = optuna.create_study(direction='maximize', study_name='ngp_study', storage='sqlite:////cephyr/users/amirmaso/Alvis/microct-neural-repr/ngp_study.db', load_if_exists=True)
        study.optimize(lambda trial: train_model(trial, start_slice, end_slice, base_img_path, opt.lr, opt.fp16, opt.workspace, opt.rhino), n_trials=50)
        # Print the best parameters.
        print(study.best_params)
        # Print the best value.
        print(study.best_value)
        # Print the best trial.
        print(study.best_trial)
        
    else:
        # Separate the dataset into training and validation data. TODO: Change so the train/validation data do not have to be a continuous range.
        # training_slices = [i for i in range(start_slice, end_slice - 2)]
        # validation_slices = [i for i in range(end_slice - 3, end_slice + 1)]

        # print(f"Training slices: {training_slices[0], training_slices[-1]}")
        # print(f"Validation slices: {validation_slices[0], validation_slices[-1]}")

        # Create the training data and load it.
        # Load data and create models
        colors, coords, H, W = load_tiff_images(start_slice, end_slice, base_img_name, resize_factor=4)
        H, W = int(H), int(W)
        # train_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=1, start_slice_index=start_slice, end_slice_index=end_slice)
        
        # H, W = train_dataset.get_H_W()
        # random_samples = H*W  # TODO: May have to cap this to the maximum number of samples that can be loaded into GPU memory.
        # train_loader = DataLoader(train_dataset, batch_size=random_samples, shuffle=True)

        # # Create the validdation data and load it
        # valid_dataset = MicroCTVolume(base_img_name=f'{base_img_path}-', resize_factor=1, start_slice_index=start_slice, end_slice_index=end_slice)
        # valid_loader  = DataLoader(valid_dataset, batch_size=random_samples)
        
        # Create dataset and dataloader for train and validation set
        num_samples = 120000
        train_dataset = MicroCTVolume(colors, coords, H, W)
        train_loader = DataLoader(train_dataset, batch_size=num_samples, shuffle=True, generator=torch.Generator(device='cpu'))

        valid_dataset = MicroCTVolume(colors[0], coords.view(colors.shape[0], H, W, 3)[0].view(-1, 3), H, W)
        valid_loader  = DataLoader(valid_dataset, batch_size=10000, generator=torch.Generator(device='cpu'))

        # Use MSELoss as criterion (loss/error function), TODO: Might need change.
        criterion = torch.nn.MSELoss()

        # Load the optimiser (Adam) with the encoding parameters (resulting hashed values when encoding) and the layers themselves.
        if opt.rhino:
            optimizer = lambda model: torch.optim.Adam([
                {'name': 'encoding', 'params': model.encoder.parameters()},
                {'name': 'transformer', 'params': model.transformer.parameters(), 'weight_decay': 1e-6},
                {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
            ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)
        else:
            optimizer = lambda model: torch.optim.Adam([
                {'name': 'encoding', 'params': model.encoder.parameters()},
                {'name': 'net', 'params': model.backbone.parameters(), 'weight_decay': 1e-6},
            ], lr=opt.lr, betas=(0.9, 0.99), eps=1e-15)

        # Create scheduler to reduce the step size after N many epochs.
        scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Initialize the trainer.
        trainer = Trainer('ngp', model, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint='latest', eval_interval=1)
        
        # Train the network
        trainer.train(train_loader, valid_loader, 10)

        # Evaluate the training
        H,W = train_dataset.get_H_W()
        trainer.test(-1, 1, (end_slice-start_slice+1), H, W, batch_size=1000)
