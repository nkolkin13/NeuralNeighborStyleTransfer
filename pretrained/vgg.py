# Core Imports
import random
import ssl

# External Dependency Imports
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import numpy as np

class Vgg16Pretrained(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16Pretrained, self).__init__()

        try:
            vgg_pretrained_features = models.vgg16(pretrained=True).features
        except ssl.SSLError:
            # unsafe fix to allow pretrained pytorch model to be downloaded
            # exposes application to man-in-the-middle attacks while model is
            # being downloaded
            create_default_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            ssl._create_default_https_context = create_default_context

        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x_in, inds=[1, 3, 6, 8, 11, 13, 15, 22, 29], concat=True):

        x = x_in.clone()  # prevent accidentally modifying input in place
        # Preprocess input according to original imagenet training
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for i in range(3):
            x[:, i:(i + 1), :, :] = (x[:, i:(i + 1), :, :] - mean[i]) / std[i]

        # Get hidden state at layers specified by 'inds'
        l2 = []
        if -1 in inds:
            l2.append(x_in)

        # Only need to run network until we get to the max depth we want outputs from
        for i in range(max(inds) + 1):
            x = self.vgg_layers[i].forward(x)
            if i in inds:
                l2.append(x)

        # Concatenate hidden states if desired (after upsampling to spatial size of largest output)
        if concat:
            if len(l2) > 1:
                zi_list = []
                max_h = l2[0].size(2)
                max_w = l2[0].size(3)
                for zi in l2:
                    if len(zi_list) == 0:
                        zi_list.append(zi)
                    else:
                        zi_list.append(F.interpolate(zi, (max_h, max_w), mode='bilinear'))

                z = torch.cat(zi_list, 1)
            else:  # don't bother doing anything if only returning one hidden state
                z = l2[0]
        else:  # Otherwise return list of hidden states
            z = l2

        return z

