# Core Dependencies
import random

# External Dependencies
import torch
import torch.nn.functional as F

# Internal Project Imports
from utils.imagePyramid import syn_lap_pyr as syn_pyr
from utils.imagePyramid import dec_lap_pyr as dec_pyr
from utils.zca import zca_tensor

def linear_2_oklab(x):
    """Converts pytorch tensor 'x' from Linear to OkLAB colorspace, described here:
        https://bottosson.github.io/posts/oklab/
    Inputs:
        x -- pytorch tensor of size B x 3 x H x W, assumed to be in linear 
             srgb colorspace, scaled between 0. and 1.
    Returns:
        y -- pytorch tensor of size B x 3 x H x W in OkLAB colorspace
    """
    assert x.size(1) == 3, "attempted to convert colorspace of tensor w/ > 3 channels"

    x = torch.clamp(x, 0., 1.)

    r = x[:, 0:1, :, :]
    g = x[:, 1:2, :, :]
    b = x[:, 2:3, :, :]

    li = 0.4121656120 * r + 0.5362752080 * g + 0.0514575653 * b
    m = 0.2118591070 * r + 0.6807189584 * g + 0.1074065790 * b
    s = 0.0883097947 * r + 0.2818474174 * g + 0.6302613616 * b

    li = torch.pow(li, 1. / 3.)
    m = torch.pow(m, 1. / 3.)
    s = torch.pow(s, 1. / 3.)

    L = 0.2104542553 * li + 0.7936177850 * m - 0.0040720468 * s
    A = 1.9779984951 * li - 2.4285922050 * m + 0.4505937099 * s
    B = 0.0259040371 * li + 0.7827717662 * m - 0.8086757660 * s

    y = torch.cat([L, A, B], 1)
    return y

def oklab_2_linear(x):
    """Converts pytorch tensor 'x' from OkLAB to Linear colorspace, described here:
        https://bottosson.github.io/posts/oklab/
    Inputs:
        x -- pytorch tensor of size B x 3 x H x W, assumed to be in OkLAB colorspace
    Returns:
        y -- pytorch tensor of size B x 3 x H x W in Linear sRGB colorspace
    """
    assert x.size(1) == 3, "attempted to convert colorspace of tensor w/ > 3 channels"

    L = x[:, 0:1, :, :]
    A = x[:, 1:2, :, :]
    B = x[:, 2:3, :, :]

    li = L + 0.3963377774 * A + 0.2158037573 * B
    m = L - 0.1055613458 * A - 0.0638541728 * B
    s = L - 0.0894841775 * A - 1.2914855480 * B

    li = torch.pow(li, 3)
    m = torch.pow(m, 3)
    s = torch.pow(s, 3)

    r = 4.0767245293 * li - 3.3072168827 * m + 0.2307590544 * s
    g = -1.2681437731 * li + 2.6093323231 * m - 0.3411344290 * s
    b = -0.0041119885 * li - 0.7034763098 * m + 1.7068625689 * s

    y = torch.cat([r, g, b], 1)
    return torch.clamp(y, 0., 1.)

def get_pad(x):
    """
    Applies 1 pixel of replication padding to x
    x -- B x D x H x W pytorch tensor
    """
    return F.pad(x, (1,1,1,1), mode='replicate')

def filter(x):
    """
    applies modified bilateral filter to AB channels of x, guided by L channel
    x -- B x 3 x H x W pytorch tensor containing an image in LAB colorspace
    """

    h = x.size(2)
    w = x.size(3)

    # Seperate out luminance channel, don't use AB channels to measure similarity
    xl = x[:,:1,:,:]
    xab = x[:,1:,:,:]
    xl_pad = get_pad(xl)

    xl_w = {}
    for i in range(3):
        for j in range(3):
            xl_w[str(i) + str(j)] =  xl_pad[:, :, i:(i+h), j:(j+w)]

    # Iteratively apply in 3x3 window rather than use spatial kernel
    max_iters = 5
    cur = torch.zeros_like(xab)

    # comparison function for pixel intensity
    def comp(x, y):
        d = torch.abs(x - y) * 5.
        return torch.pow(torch.exp(-1. * d),2)

    # apply bilateral filtering to AB channels, guideded by L channel 
    cur = xab.clone()
    for it in range(max_iters):
        cur_pad = get_pad(cur)
        xl_v = {}
        for i in range(3):
            for j in range(3):
                xl_v[str(i) + str(j)] = cur_pad[:, :, i:(i+h), j:(j+w)]

        denom = torch.zeros_like(xl)
        cur = cur * 0.

        for i in range(3):
            for j in range(3):
                scl = comp(xl, xl_w[str(i) + str(j)])
                cur = cur + xl_v[str(i) + str(j)] * scl
                denom = denom + scl

        cur = cur / denom
    # store result and return
    x[:, 1:, :, :] = cur
    return x

def clamp_range(x, y):
    '''
    clamp the range of x to [min(y), max(y)]
    x -- pytorch tensor
    y -- pytorch tensor
    '''
    return torch.clamp(x, y.min(), y.max())

def color_match(c, s, o, moment_only=False):
    '''
    Constrain the low frequences of the AB channels of output image 'o' (containing hue and saturation)
    to be an affine transformation of 'c' matching the mean and covariance of the style image 's'.
    Compared to the raw output of optimization this is highly constrained, but in practice
    we find the benefit to robustness to be worth the reduced stylization.
    c -- B x 3 x H x W pytorch tensor containing content image
    s -- B x 3 x H x W pytorch tensor containing style image
    o -- B x 3 x H x W pytorch tensor containing initial output image
    moment_only -- boolean, prevents applying bilateral filter to AB channels of final output to match luminance's edges
    '''
    c = torch.clamp(c, 0., 1.)
    s = torch.clamp(s, 0., 1.)
    o = torch.clamp(o, 0., 1.)

    x = linear_2_oklab(c)
    x_flat = x.view(x.size(0), x.size(1), -1, 1)
    y = linear_2_oklab(s)
    o = linear_2_oklab(o)

    x_new = o.clone()
    for i in range(3):
        x_new[:, i:i + 1,:,:] = clamp_range(x_new[:, i:i + 1,:,:], y[:, i:i + 1, :, :])

    _, cov_s = zca_tensor(x_new, y)

    if moment_only or cov_s[1:,1:].abs().max() < 6e-5:
       x_new[:,1:,:,:] = o[:,1:,:,:]
       x_new, _ = zca_tensor(x_new, y)
    else:
        x_new[:,1:,:,:] = x[:,1:,:,:]
        x_new[:,1:,:,:] = zca_tensor(x_new[:,1:,:,:], y[:,1:,:,:])[0]
        x_new = filter(x_new)

    for i in range(3):
        x_new[:,i:i+1,:,:] = clamp_range(x_new[:,i:i+1,:,:], y[:,i:i+1,:,:])

    x_pyr = dec_pyr(x,4)
    y_pyr = dec_pyr(y,4)
    x_new_pyr = dec_pyr(x_new,4)
    o_pyr = dec_pyr(o,4)
    x_new_pyr[:-1] = o_pyr[:-1]

    return oklab_2_linear(x_new)
