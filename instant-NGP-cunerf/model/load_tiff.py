# Import necessary libraries
import tifffile as tiff
import torch
import torch.nn as nn
import numpy as np

from scipy.spatial.transform import Rotation as R


from torchvision.transforms import Compose, ToTensor, Resize, Normalize, InterpolationMode

def get_mgrid(x_dim, y_dim, z_dim):

    tensors = tuple([torch.linspace(0, x_dim, x_dim)] + [torch.linspace(0, y_dim, y_dim)] + [torch.linspace(0, z_dim, z_dim)])
    grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 3)

    grid = normalize(grid, x_dim, y_dim, z_dim)

    grid = grid[grid[:, 0].argsort()]
    # indices = np.lexsort((grid[:, 0], grid[:, 1], grid[:, 2]))
    indices = np.lexsort((grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy(), grid[:, 2].cpu().numpy()))
    grid = grid[indices]

    return grid

def get_slice_mgrid(x_dim, y_dim, z):

    tensors = tuple([torch.linspace(0, x_dim, x_dim)] + [torch.linspace(0, y_dim, y_dim)] + [torch.tensor(float(z))])
    grid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    grid = grid.reshape(-1, 3)

    grid = normalize(grid, x_dim, y_dim, 1)

    grid = grid[grid[:, 0].argsort()]
    indices = np.lexsort((grid[:, 0].cpu().numpy(), grid[:, 1].cpu().numpy()))
    grid = grid[indices]

    return grid

def get_view_mgrid(x_dim, y_dim, translation, rotation_angles):

    # Generate translation matrix for the given translation vector
    translation_matrix = torch.eye(4)
    translation_matrix[0, 3] = translation[0]
    translation_matrix[1, 3] = translation[1]
    translation_matrix[2, 3] = translation[2]

    # Generate rotation matrix for the given rotation vector given in euler angles
    rotation_matrix = R.from_euler('xyz', rotation_angles).as_matrix()
    rotation_matrix = torch.from_numpy(rotation_matrix)

    # Convert the 3x3 rotation matrix to a 4x4 matrix
    rotation_matrix_4x4 = torch.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    # Generate the view matrix
    transformation_matrix = torch.matmul(translation_matrix, rotation_matrix_4x4)

    # Generate the coordinates for the new view by multiplying the transformation matrix with the middle slice coordinates
    p0 = get_slice_mgrid(x_dim, y_dim, 1/2)

    # Convert p0 to homogeneous coordinates
    p0_homogeneous = torch.cat([p0, torch.ones(p0.shape[0], 1)], dim=1).T

    p_new = torch.matmul(transformation_matrix, p0_homogeneous)

    # Convert p_new back to 3D coordinates
    p_new = p_new[:3, :].T

    return p_new


def normalize(coordinates, x_dim, y_dim, z_dim):
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    x_norm = 2*(x-(x_dim/2)) / x_dim
    y_norm = 2*(y-(y_dim/2)) / y_dim
    z_norm = 2*(z-(z_dim/2)) / z_dim

    norm_coordinates = torch.stack([x_norm, y_norm, z_norm], dim=1)
    return norm_coordinates

def load_tiff_images(start_index, end_index, base_img_name, resize_factor=1):

    colors = []
    for slice_img in range(start_index, end_index + 1):
        imgpath = base_img_name + "-" + '{:04d}'.format(slice_img) + '.tif'
        img = tiff.imread(imgpath)

        img = img.astype('float32')

        # Normalize pixel values between 0 and 1
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

        H, W = int(img.shape[0] / resize_factor), int(img.shape[1] / resize_factor)
        transform = Compose([
            ToTensor(),
            Resize((H, W), interpolation=InterpolationMode.NEAREST)
        ])
        img = transform(img)[0]
        colors.append(img)

    colors = torch.stack(colors)
    coords = get_mgrid(H, W, (end_index + 1)-start_index)
    H, W = colors[0].shape

    return colors, coords, H, W


# if __name__ == "__main__":
#     img = np.zeros((3, 3, 3))

#     for i in range(3):
#         img[i] = np.arange(i*9+1, i*9+10).reshape(3, 3)

#     # Save the images
#     for i in range(3):
#         tiff.imwrite(f'test{i}.tif', img[i])

#     # Load the images using load_tiff_images
#     slices, coords, H, W = load_tiff_images(0, 2, 'test')

#     # Print the loaded images
#     slices = slices.reshape(-1, 1)
#     coords = coords.reshape(-1, 3)
#     pixels = torch.tensor([[coords[0], coords[1], coords[2], color] for coords, color in zip(coords, slices)])

#     print("Generating new slices for the given view. should give the middle slice coords")
#     new_coords = get_view_mgrid(H, W, [0, 0, 0], [0, 0, 0])

#     print(new_coords)




