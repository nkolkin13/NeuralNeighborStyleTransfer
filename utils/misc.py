# Core Imports
import subprocess

# External Dependency Imports
import numpy as np
import torch
import torch.nn.functional as F
from imageio import imread

# global variable
USE_GPU = True

def to_device(tensor):
    """Ensures torch tensor 'tensor' is moved to gpu
    if global variable USE_GPU is True"""
    if USE_GPU:
        return tensor.cuda()
    else:
        return tensor

def match_device(ref, mut):
    """ Puts torch tensor 'mut' on the same device as torch tensor 'ref'"""
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

def get_gpu_memory_map():
    """Get the current gpu usage. Taken from:
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)

def flatten_grid(x):
    """ collapses spatial dimensions of pytorch tensor 'x' and transposes
        Inputs:
            x -- 1xCxHxW pytorch tensor
        Outputs:
            y -- (H*W)xC pytorch tensor
    """
    assert x.size(0) == 1, "undefined behavior for batched input"
    y = x.contiguous().view(x.size(1), -1).clone().transpose(1, 0)
    return y

def scl_spatial(x, h, w):
    """shorter alias for default way I call F.interpolate (i.e. as bilinear
    interpolation
    """
    return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

def load_path_for_pytorch(im_path, target_size=1000, side_comp=max, verbose=False):
    """
    Loads image at 'path', selects height or width with function 'side_comp'
    then scales the image, setting selected dimension to 'target_size' and
    maintaining aspect ratio. Will also convert RGBA or greyscale images to
    RGB

    Returns:
        x -- a HxWxC pytorch tensor of rgb values scaled between 0. and 1.
    """
    # Load Image
    x = imread(im_path).astype(np.float32)


    # Converts image to rgb if greyscale
    if len(x.shape) < 3:
        x = np.stack([x, x, x], 2)

    # Removes alpha channel if present
    if x.shape[2] > 3:
        x = x[:, :, :3]

    # Rescale rgb values
    x = x / 255.

    # Convert from numpy
    x_dims = x.shape
    x = torch.from_numpy(x).contiguous().permute(2, 0, 1).contiguous()

    # Rescale to desired size
    # by default maintains aspect ratio relative to long side
    # change side_comp to be min for short side
    fac = float(target_size) / side_comp(x_dims[:2])
    h = int(x_dims[0] * fac)
    w = int(x_dims[1] * fac)
    x = scl_spatial(x.unsqueeze(0), h, w)[0]

    if verbose:
        print(f'DEBUG: image from path {im_path} loaded with size {x_dims}')

    return x
