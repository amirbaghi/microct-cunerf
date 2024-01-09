import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim

from ngp_network_RHINO import INGPNetworkRHINO
from ngp_network import INGPNetwork
from utils import *
from load_tiff import load_tiff_images
from skimage.metrics import structural_similarity
from torchmetrics.functional import peak_signal_noise_ratio

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
    parser.add_argument('--translation', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="translation")
    parser.add_argument('--rotation_angle', type=float, default=0.0, help="rotation angles")
    parser.add_argument('--rotation_axis', type=float, nargs='+', default=[1.0, 0.0, 0.0], help="rotation axis")

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
        # coarse_model = INGPNetworkRHINO(num_layers=9, hidden_dim=128, skips=[4, 7], input_dim=3, num_levels=8,
        #                 level_dim=2, base_resolution=16, log2_hashmap_size=13, desired_resolution=H, 
        #                 align_corners=False, freq=40, transformer_num_layers=1, transformer_hidden_dim=32)
        # fine_model = INGPNetworkRHINO(num_layers=9, hidden_dim=256, skips=[4, 7], input_dim=3, num_levels=16,
        #         level_dim=4, base_resolution=8, log2_hashmap_size=17, desired_resolution=2*H, 
        #         align_corners=False, freq=50, transformer_num_layers=1, transformer_hidden_dim=32)

        # # Use this! cunerf-rhino-resize4
        # coarse_model = INGPNetworkRHINO(num_layers=9, hidden_dim=128, skips=[4, 7], input_dim=3, num_levels=8,
        #                 level_dim=2, base_resolution=16, log2_hashmap_size=10, desired_resolution=H, 
        #                 align_corners=False, freq=20, transformer_num_layers=1, transformer_hidden_dim=32)
        # fine_model = INGPNetworkRHINO(num_layers=9, hidden_dim=256, skips=[4, 7], input_dim=3, num_levels=16,
        #         level_dim=4, base_resolution=16, log2_hashmap_size=16, desired_resolution=2*H, 
        #         align_corners=False, freq=50, transformer_num_layers=1, transformer_hidden_dim=128)

       # Use this! cunerf-rhino-2-r4
        coarse_model = INGPNetworkRHINO(num_layers=9, hidden_dim=128, skips=[4, 7], input_dim=3, num_levels=16,
                        level_dim=2, base_resolution=16, log2_hashmap_size=15, desired_resolution=H, 
                        align_corners=False, freq=20, transformer_num_layers=1, transformer_hidden_dim=32)
        fine_model = INGPNetworkRHINO(num_layers=9, hidden_dim=256, skips=[4, 7], input_dim=3, num_levels=16,
                level_dim=2, base_resolution=16, log2_hashmap_size=12, desired_resolution=2*H, 
                align_corners=False, freq=50, transformer_num_layers=1, transformer_hidden_dim=128)
    else:
        model = INGPNetwork(num_layers=5, hidden_dim=512, input_dim=3, num_levels=17, 
                        level_dim=4, base_resolution=16, log2_hashmap_size=21, desired_resolution=261, 
                        align_corners=False)


    dataset_dir = find_directory('dataset')
    if dataset_dir is None:
        print("Dataset directory not found. Please download the dataset as described in README.md.")
        exit(1)
    
    base_img_path = os.path.join(dataset_dir, base_img_name)

    if opt.test:
        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)
        H, W = int(H), int(W)
        trainer.test(0, 1, (end_slice-start_slice+1), H, W, batch_size=1000)
        trainer.test(0, 1, 2 * (end_slice-start_slice+1), H, W, batch_size=1000)

    elif opt.test_on_slice:
        print(f'Testing on slice {opt.test_slice}...')

        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)

        slice_index = opt.test_slice
        slice_colors = colors[slice_index]
        slice_coords = coords.view(colors.shape[0], H, W, 3)[slice_index].view(-1, 3)
        prediction = trainer.test_image_partial(H, W, slice_coords, 'cuda', batch_size=1000, imagepath=f'slice_{slice_index:.2f}_pred.png')

        # Save the ground truth
        slice_colors = slice_colors.view(H, W).numpy()
        slice_colors = (slice_colors * 65535).astype(np.uint16)
        slice_colors = np.array(slice_colors, dtype=np.uint16)
        plt.imsave(f'gt_{slice_index}.png', slice_colors, cmap="gray")

        # Calculate PSNR and SSIM
        # Calculate loss between the ground truth and the predictions
        prediction = prediction.cpu()
        slice_colors = slice_colors.flatten().cpu()

        mse = torch.mean((slice_colors - prediction) ** 2)
        psnr = peak_signal_noise_ratio(slice_colors, prediction)
        # ssim = structural_similarity(slice_colors.numpy().astype(np.float64), prediction.numpy().astype(np.float64), data_range=prediction.max() - prediction.min())

        print(f'Reconstructed slice with MSE: {mse:.5f}, PSNR: {psnr:.2f}')

    elif opt.render_new_view:
        print(f'Rendering new view...')

        trainer = Trainer('ngp', coarse_model, fine_model, workspace=opt.workspace, ema_decay=0.95, fp16=opt.fp16, use_checkpoint='latest',
                         eval_interval=1, length=cube_lengths, num_cube_samples=num_coarse_samples, num_fine_samples=num_fine_samples)

        for x_rot in range(0, 360, 10):
            translation = opt.translation
            rotation_angle = x_rot
            rotation_axis = opt.rotation_axis
            view_coords = get_view_mgrid(H, W, translation, rotation_angle, rotation_axis)

            prediction = trainer.test_image_partial(H, W, view_coords, 'cuda', batch_size=1000, imagepath=f'new_view_{x_rot}.png')

        # translation = opt.translation
        # rotation_angle = opt.rotation_angle
        # rotation_axis = opt.rotation_axis
        # view_coords = get_view_mgrid(H, W, translation, rotation_angle, rotation_axis)

        # prediction = trainer.test_image_partial(H, W, view_coords, 'cuda', batch_size=1000, imagepath=f'new_view.png')

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
        trainer.test(0, (end_slice-start_slice+1), (end_slice-start_slice+1), H, W, batch_size=1000)
