# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=2,
        bias=True)


def deconv4x4(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=4, stride=stride, padding=1)


def deconv5x5(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=2,
        output_padding=1)


def actFunc(act, *args, **kwargs):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'relu6':
        return nn.ReLU6()
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.1)
    elif act == 'prelu':
        return nn.PReLU()
    elif act == 'rrelu':
        return nn.RReLU(0.1, 0.3)
    elif act == 'selu':
        return nn.SELU()
    elif act == 'celu':
        return nn.CELU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError


def make_blocks(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)
