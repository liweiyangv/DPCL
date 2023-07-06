import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Memory import *


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class Encoder(torch.nn.Module):
    def __init__(self, t_length=2, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )
        n_blocks = 9
        model_encoder = []
        norm_layer = get_norm_layer(norm_type='instance')
        for i in range(n_blocks):  # add ResNet blocks
            model_encoder += [
                ResnetBlock(512, padding_type='reflect', norm_layer=norm_layer, use_dropout=False,
                            use_bias=True)]

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        self.resnetlayer = nn.Sequential(*model_encoder)



    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorresnetbloack = self.resnetlayer(tensorConv4)
        return tensorresnetbloack


class Decoder(torch.nn.Module):
    def __init__(self, t_length=2, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.InstanceNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(1024, 512)
        self.moduleUpsample4 = Upsample(512, 512)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 256)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 128)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

    def forward(self, x):
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)

        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        output = self.moduleDeconv1(tensorUpsample2)

        return output


class convAE(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=2, memory_size=10, feature_dim=512, key_dim=512, temp_update=0.1,
                 temp_gather=0.1):
        super(convAE, self).__init__()

        self.encoder = Encoder(t_length, n_channel)
        self.decoder = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size, feature_dim, key_dim, temp_update, temp_gather)

    def forward(self, x, keys, train=True):

        fea = self.encoder(x)
        if train:
            updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, keys, train)
            output = self.decoder(updated_fea)

            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss

        # test
        else:
            # updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss = self.memory(fea, keys, train)
            # output = self.decoder(updated_fea)
            #
            # return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss
            updated_fea, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(
                fea, keys, train)
            output = self.decoder(updated_fea)

            return output, fea, updated_fea, keys, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss




