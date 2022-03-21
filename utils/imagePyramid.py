# External Dependency Imports
import torch.nn.functional as F

def dec_lap_pyr(x, levs):
    """ constructs batch of 'levs' level laplacian pyramids from x
        Inputs:
            x -- BxCxHxW pytorch tensor
            levs -- integer number of pyramid levels to construct
        Outputs:
            pyr -- a list of pytorch tensors, each representing a pyramid level,
                   pyr[0] contains the finest level, pyr[-1] the coarsest
    """
    pyr = []
    cur = x  # Initialize approx. coefficients with original image
    for i in range(levs):

        # Construct and store detail coefficients from current approx. coefficients
        h = cur.size(2)
        w = cur.size(3)
        x_small = F.interpolate(cur, (h // 2, w // 2), mode='bilinear')
        x_back = F.interpolate(x_small, (h, w), mode='bilinear')
        lap = cur - x_back
        pyr.append(lap)

        # Store new approx. coefficients
        cur = x_small

    pyr.append(cur)

    return pyr

def syn_lap_pyr(pyr):
    """ collapse batch of laplacian pyramids stored in list of pytorch tensors
        'pyr' into a single tensor.
        Inputs:
            pyr -- list of pytorch tensors, where pyr[i] has size BxCx(H/(2**i)x(W/(2**i))
        Outpus:
            x -- a BxCxHxW pytorch tensor
    """
    cur = pyr[-1]
    levs = len(pyr)

    for i in range(0, levs - 1)[::-1]:
        # Create new approximation coefficients from current approx. and detail coefficients
        # at next finest pyramid level
        up_x = pyr[i].size(2)
        up_y = pyr[i].size(3)
        cur = pyr[i] + F.interpolate(cur, (up_x, up_y), mode='bilinear')
    x = cur

    return x
