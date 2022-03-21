# Core Dependencies
import random

# External Dependency Imports
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Internal Project Imports
from utils.imagePyramid import syn_lap_pyr as syn_pyr
from utils.imagePyramid import dec_lap_pyr as dec_pyr
from utils.distance import pairwise_distances_l2, pairwise_distances_cos_center
from utils.featureExtract import extract_feats, get_feat_norms
from utils import misc
from utils.misc import to_device, flatten_grid, scl_spatial
from utils.colorization import color_match

def produce_stylization(content_im, style_im, phi,
                        max_iter=350,
                        lr=1e-3,
                        content_weight=1.,
                        max_scls=0,
                        flip_aug=False,
                        content_loss=False,
                        zero_init=False,
                        dont_colorize=False):
    """ Produce stylization of 'content_im' in the style of 'style_im'
        Inputs:
            content_im -- 1x3xHxW pytorch tensor containing rbg content image
            style_im -- 1x3xH'xW' pytorch tensor containing rgb style image
            phi -- lambda function to extract features using VGG16Pretrained
            max_iter -- number of updates to image pyramid per scale
            lr -- learning rate of optimizer updating pyramid coefficients
            content_weight -- controls stylization level, between 0 and 1
            max_scl -- number of scales to stylize (performed coarse to fine)
            flip_aug -- extract features from rotations of style image too?
            content_loss -- use self-sim content loss? (compares downsampled
                            version of output and content image)
            zero_init -- if true initialize w/ grey image, o.w. initialize w/
                         downsampled content image
    """
    # Get max side length of final output and set number of pyramid levels to 
    # optimize over
    max_size = max(content_im.size(2), content_im.size(3))
    pyr_levs = 8

    # Decompose style image, content image, and output image into laplacian 
    # pyramid
    style_pyr = dec_pyr(style_im, pyr_levs)
    c_pyr = dec_pyr(content_im, pyr_levs)
    s_pyr = dec_pyr(content_im.clone(), pyr_levs)

    # Initialize output image pyramid
    if zero_init:
        # Initialize with flat grey image (works, but less vivid)
        for i in range(len(s_pyr)):
            s_pyr[i] = s_pyr[i] * 0.
        s_pyr[-1] = s_pyr[-1] * 0. + 0.5

    else:
        # Initialize with low-res version of content image (generally better 
        # results, improves contrast of final output)
        z_max = 2
        if max_size < 1024:
            z_max = 3

        for i in range(z_max):
            s_pyr[i] = s_pyr[i] * 0.

    # Stylize using hypercolumn matching from coarse to fine scale
    li = -1
    for scl in range(max_scls)[::-1]:

        # Get content image and style image from pyramid at current resolution
        if misc.USE_GPU:
            torch.cuda.empty_cache()
        style_im_tmp = syn_pyr(style_pyr[scl:])
        content_im_tmp = syn_pyr(c_pyr[scl:])
        output_im_tmp = syn_pyr(s_pyr[scl:])
        li += 1
        print(f'-{li, max(output_im_tmp.size(2),output_im_tmp.size(3))}-')


        # Construct stylized activations
        with torch.no_grad():

            # Control tradeoff between searching for features that match
            # current iterate, and features that match content image (at
            # coarsest scale, only use content image)    
            alpha = content_weight
            if li == 0:
                alpha = 0.

            # Search for features using high frequencies from content 
            # (but do not initialize actual output with them)
            output_extract = syn_pyr([c_pyr[scl]] + s_pyr[(scl + 1):])

            # Extract style features from rotated copies of style image
            feats_s = extract_feats(style_im_tmp, phi, flip_aug=flip_aug).cpu()

            # Extract features from convex combination of content image and
            # current iterate:
            c_tmp = (output_extract * alpha) + (content_im_tmp * (1. - alpha))
            feats_c = extract_feats(c_tmp, phi).cpu()

            # Replace content features with style features
            target_feats = replace_features(feats_c, feats_s)

        # Synthesize output at current resolution using hypercolumn matching
        s_pyr = optimize_output_im(s_pyr, c_pyr, content_im, style_im_tmp,
                                   target_feats, lr, max_iter, scl, phi,
                                   content_loss=content_loss)

        # Get output at current resolution from pyramid
        with torch.no_grad():
            output_im = syn_pyr(s_pyr)

    # Perform final pass using feature splitting (pass in flip_aug argument
    # because style features are extracted internally in this regime)
    s_pyr = optimize_output_im(s_pyr, c_pyr, content_im, style_im_tmp,
                               target_feats, lr, max_iter, scl, phi,
                               final_pass=True, content_loss=content_loss,
                               flip_aug=flip_aug)

    # Get final output from pyramid
    with torch.no_grad():
        output_im = syn_pyr(s_pyr)

    if dont_colorize:
        return output_im
    else:
        return color_match(content_im, style_im, output_im)

def replace_features(src, ref):
    """ Replace each feature vector in 'src' with the nearest (under centered 
    cosine distance) feature vector in 'ref'
    Inputs:
        src -- 1xCxAxB tensor of content features
        ref -- 1xCxHxW tensor of style features
    Outputs:
        rplc -- 1xCxHxW tensor of features, where rplc[0,:,i,j] is the nearest
                neighbor feature vector of src[0,:,i,j] in ref
    """
    # Move style features to gpu (necessary to mostly store on cpu for gpus w/
    # < 12GB of memory)
    ref_flat = to_device(flatten_grid(ref))
    rplc = []
    for j in range(src.size(0)):
        # How many rows of the distance matrix to compute at once, can be
        # reduced if less memory is available, but this slows method down
        stride = 128**2 // max(1, (ref.size(2) * ref.size(3)) // (128 ** 2))
        bi = 0
        ei = 0

        # Loop until all content features are replaced by style feature / all
        # rows of distance matrix are computed
        out = []
        src_flat_all = flatten_grid(src[j:j + 1, :, :, :])
        while bi < src_flat_all.size(0):
            ei = min(bi + stride, src_flat_all.size(0))

            # Get chunck of content features, compute corresponding portion
            # distance matrix, and store nearest style feature to each content
            # feature
            src_flat = to_device(src_flat_all[bi:ei, :])
            d_mat = pairwise_distances_cos_center(ref_flat, src_flat)
            _, nn_inds = torch.min(d_mat, 0)
            del d_mat  # distance matrix uses lots of memory, free asap

            # Get style feature closest to each content feature and save
            # in 'out'
            nn_inds = nn_inds.unsqueeze(1).expand(nn_inds.size(0), ref_flat.size(1))
            ref_sel = torch.gather(ref_flat, 0, nn_inds).transpose(1,0).contiguous()
            out.append(ref_sel)#.view(1, ref.size(1), src.size(2), ei - bi))

            bi = ei

        out = torch.cat(out, 1)
        out = out.view(1, ref.size(1), src.size(2), src.size(3))
        rplc.append(out)

    rplc = torch.cat(rplc, 0)
    return rplc

def optimize_output_im(s_pyr, c_pyr, content_im, style_im, target_feats,
                       lr, max_iter, scl, phi, final_pass=False,
                       content_loss=False, flip_aug=True):
    ''' Optimize laplacian pyramid coefficients of stylized image at a given
        resolution, and return stylized pyramid coefficients.
        Inputs:
            s_pyr -- laplacian pyramid of style image
            c_pyr -- laplacian pyramid of content image
            content_im -- content image
            style_im -- style image
            target_feats -- precomputed target features of stylized output
            lr -- learning rate for optimization
            max_iter -- maximum number of optimization iterations
            scl -- integer controls which resolution to optimize (corresponds
                   to pyramid level of target resolution)
            phi -- lambda function to compute features using pretrained VGG16
            final_pass -- if true, ignore 'target_feats' and recompute target
                          features before every step of gradient descent (and
                          compute feature matches seperately for each layer
                          instead of using hypercolumns)
            content_loss -- if true, also minimize content loss that maintains
                            self-similarity in color space between 32pixel
                            downsampled output image and content image
            flip_aug -- if true, extract style features from rotations of style
                        image. This increases content preservation by making
                        more options available when matching style features
                        to content features
        Outputs:
            s_pyr -- pyramid coefficients of stylized output image at target
                     resolution
    '''
    # Initialize optimizer variables and optimizer       
    output_im = syn_pyr(s_pyr[scl:])
    opt_vars = [Variable(li.data, requires_grad=True) for li in s_pyr[scl:]]
    optimizer = torch.optim.Adam(opt_vars, lr=lr)

    # Original features uses all layers, but dropping conv5 block  speeds up 
    # method without hurting quality
    feature_list_final = [22, 20, 18, 15, 13, 11, 8, 6, 3, 1]

    # Precompute features that remain constant
    if not final_pass:
        # Precompute normalized features targets during hypercolumn-matching 
        # regime for cosine distance
        target_feats_n = target_feats / get_feat_norms(target_feats)

    else:
        # For feature-splitting regime extract style features for each conv 
        # layer without downsampling (including from rotations if applicable)
        s_feat = phi(style_im, feature_list_final, False)

        if flip_aug:
            aug_list = [torch.flip(style_im, [2]).transpose(2, 3),
                        torch.flip(style_im, [2, 3]),
                        torch.flip(style_im, [3]).transpose(2, 3)]

            for ia, im_aug in enumerate(aug_list):
                s_feat_tmp = phi(im_aug, feature_list_final, False)

                if ia != 1:
                    s_feat_tmp = [s_feat_tmp[iii].transpose(2, 3)
                                  for iii in range(len(s_feat_tmp))]

                s_feat = [torch.cat([s_feat[iii], s_feat_tmp[iii]], 2)
                          for iii in range(len(s_feat_tmp))]

    # Precompute content self-similarity matrix if needed for 'content_loss'
    if content_loss:
        c_full = syn_pyr(c_pyr)
        c_scl = max(c_full.size(2), c_full.size(3))
        c_fac = c_scl // 32
        h = int(c_full.size(2) / c_fac)
        w = int(c_full.size(3) / c_fac)

        c_low_flat = flatten_grid(scl_spatial(c_full, h, w))
        self_sim_target = pairwise_distances_l2(c_low_flat, c_low_flat).clone().detach()


    # Optimize pyramid coefficients to find image that produces stylized activations
    for i in range(max_iter):

        # Zero out gradient and loss before current iteration
        optimizer.zero_grad()
        ell = 0.

        # Synthesize current output from pyramid coefficients
        output_im = syn_pyr(opt_vars)


        # Compare current features with stylized activations
        if not final_pass:  # hypercolumn matching / 'hm' regime

            # Extract features from current output, normalize for cos distance
            cur_feats = extract_feats(output_im, phi)
            cur_feats_n = cur_feats / get_feat_norms(cur_feats)

            # Update overall loss w/ cosine loss w.r.t target features
            ell = ell + (1. - (target_feats_n * cur_feats_n).sum(1)).mean()


        else:  # feature splitting / 'fs' regime
            # Extract features from current output (keep each layer seperate 
            # and don't downsample)
            cur_feats = phi(output_im, feature_list_final, False)

            # Compute matches for each layer. For efficiency don't explicitly 
            # gather matches, only access through distance matrix.
            ell_fs = 0.
            for h_i in range(len(s_feat)):
                # Get features from a particular layer
                s_tmp = s_feat[h_i]
                cur_tmp = cur_feats[h_i]
                chans = s_tmp.size(1)

                # Sparsely sample feature tensors if too big, otherwise just 
                # reshape
                if max(cur_tmp.size(2), cur_tmp.size(3)) > 64:
                    stride = max(cur_tmp.size(2), cur_tmp.size(3)) // 64
                    offset_a = random.randint(0, stride - 1)
                    offset_b = random.randint(0, stride - 1)
                    s_tmp = s_tmp[:, :, offset_a::stride, offset_b::stride]
                    cs_tmp = cur_tmp[:, :, offset_a::stride, offset_b::stride]

                r_col_samp = s_tmp.contiguous().view(1, chans, -1)
                s_col_samp = cs_tmp.contiguous().view(1, chans, -1)

                # Compute distance matrix and find minimum along each row to 
                # implicitly get matches (and minimize distance between them)
                d_mat = pairwise_distances_cos_center(r_col_samp[0].transpose(1, 0),
                                                      s_col_samp[0].transpose(1, 0))
                d_min, _ = torch.min(d_mat, 0)

                # Aggregate loss over layers
                ell_fs = ell_fs + d_min.mean()

            # Update overall loss
            ell = ell + ell_fs

        # Optional self similarity content loss between downsampled output 
        # and content image. Always turn off at end for best results.
        if content_loss and not (final_pass and i > 100):
            o_scl = max(output_im.size(2), output_im.size(3))
            o_fac = o_scl / 32.
            h = int(output_im.size(2) / o_fac)
            w = int(output_im.size(3) / o_fac)

            o_flat = flatten_grid(scl_spatial(output_im, h, w))
            self_sim_out = pairwise_distances_l2(o_flat, o_flat)

            ell = ell + torch.mean(torch.abs((self_sim_out - self_sim_target)))

        # Update output's pyramid coefficients
        ell.backward()
        optimizer.step()

    # Update output's pyramid coefficients for current resolution
    # (and all coarser resolutions)    
    s_pyr[scl:] = dec_pyr(output_im, len(c_pyr) - 1 - scl)
    return s_pyr
