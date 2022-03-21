# External Dependency Imports
import torch
import torch.nn.functional as F

# Internal Project Imports
from utils.misc import scl_spatial

def get_feat_norms(x):
    """ Makes l2 norm of x[i,:,j,k] = 1 for all i,j,k. Clamps before sqrt for
    stability
    """
    return torch.clamp(x.pow(2).sum(1, keepdim=True), 1e-8, 1e8).sqrt()


def phi_cat(x, phi, layer_l):
    """ Extract conv features from 'x' at list of VGG16 layers 'layer_l'. Then
        normalize features from each conv block based on # of channels, resize,
        and concatenate into hypercolumns
        Inputs:
            x -- Bx3xHxW pytorch tensor, presumed to contain rgb images
            phi -- lambda function calling a pretrained Vgg16Pretrained model
            layer_l -- layer indexes to form hypercolumns out of
        Outputs:
            feats -- BxCxHxW pytorch tensor of hypercolumns extracted from 'x'
                     C depends on 'layer_l'
    """
    h = x.size(2)
    w = x.size(3)

    feats = phi(x, layer_l, False)
    # Normalize each layer by # channels so # of channels doesn't dominate 
    # cosine distance
    feats = [f / f.size(1) for f in feats]

    # Scale layers' features to target size and concatenate
    feats = torch.cat([scl_spatial(f, h // 4, w // 4) for f in feats], 1) 

    return feats

def extract_feats(im, phi, flip_aug=False):
    """ Extract hypercolumns from 'im' using pretrained VGG16 (passed as phi),
    if speficied, extract hypercolumns from rotations of 'im' as well
        Inputs:
            im -- a Bx3xHxW pytorch tensor, presumed to contain rgb images
            phi -- a lambda function calling a pretrained Vgg16Pretrained model
            flip_aug -- whether to extract hypercolumns from rotations of 'im'
                        as well
        Outputs:
            feats -- a tensor of hypercolumns extracted from 'im', spatial
                     index is presumed to no longer matter
    """
    # In the original paper used all layers, but dropping conv5 block increases
    # speed without harming quality
    layer_l = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]
    feats = phi_cat(im, phi, layer_l)

    # If specified, extract features from 90, 180, 270 degree rotations of 'im'
    if flip_aug:
        aug_list = [torch.flip(im, [2]).transpose(2, 3),
                    torch.flip(im, [2, 3]),
                    torch.flip(im, [3]).transpose(2, 3)]

        for i, im_aug in enumerate(aug_list):
            feats_new = phi_cat(im_aug, phi, layer_l)

            # Code never looks at patches of features, so fine to just stick
            # features from rotated images in adjacent spatial indexes, since
            # they will only be accessed in isolation
            if i == 1:
                feats = torch.cat([feats, feats_new], 2)
            else:
                feats = torch.cat([feats, feats_new.transpose(2, 3)], 2)

    return feats
