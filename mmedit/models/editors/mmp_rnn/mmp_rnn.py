# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmedit.registry import MODELS
from .mmp_rnn_unet import UNet
from .mmp_rnn_utils import actFunc, conv1x1, conv3x3, conv5x5


@MODELS.register_module()
class MMPRNN(nn.Module):
    """MMPRNN module."""

    def __init__(self, para):
        super(MMPRNN, self).__init__()
        self.para = para
        self.n_feats = para['n_features']
        self.num_ff = para['future_frames']
        self.num_fb = para['past_frames']
        self.ds_ratio = 4
        self.cell = RDBCell(para)
        self.recons = Reconstructor(para)
        self.fusion = CAT(para)
        self.do_skip = para['do_skip']
        self.centralize = para['centralize']
        self.normalize = para['normalize']
        if self.do_skip is True:
            self.skip = nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=9,
                stride=1,
                padding=4,
                bias=True)

    def forward(self, x):
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        s = torch.zeros(batch_size, self.n_feats, s_height,
                        s_width).to(x.device)
        mid = torch.zeros(batch_size, 2 * self.n_feats, s_height,
                          s_width).to(x.device)
        for i in range(frames):
            h, s, mid = self.cell(x[:, i, :, :, :], s, mid)
            hs.append(h)
        for i in range(self.num_fb, frames - self.num_ff):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])

            out = self.recons(out)
            if self.do_skip is True:
                skip = self.skip(x[:, i, 0:3, :, :])
                out = out + skip
            outputs.append(out.unsqueeze(dim=1))
        return torch.cat(outputs, dim=1).squeeze()


class RDBCell(nn.Module):

    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para['activation']
        self.n_feats = para['n_features']
        self.n_blocks = para['n_blocks_a']
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.MMAM1 = MMAMLayer(para)

        self.F_B1 = RDB_DS(
            in_channels=1 * self.n_feats,
            growthRate=int(self.n_feats * 2 / 2),
            out_channels=2 * self.n_feats,
            num_layer=3,
            activation=self.activation)
        self.F_B2 = RDB_DS(
            in_channels=2 * self.n_feats,
            growthRate=int(self.n_feats * 3 / 2),
            out_channels=2 * self.n_feats,
            num_layer=3,
            activation=self.activation)
        self.F_R = RDNet(
            in_channels=(1 + 2 + 2) * self.n_feats,
            growthRate=2 * self.n_feats,
            num_layer=3,
            num_blocks=self.n_blocks,
            activation=self.activation,
            do_attention_ca=False,
            do_attention_sa=False)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3((1 + 2 + 2) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats,
                growthRate=self.n_feats,
                num_layer=3,
                activation=self.activation), conv3x3(self.n_feats,
                                                     self.n_feats))

        self.prior = UNet()
        checkpoint = torch.load('checkpoint/mmp_unet.pth.tar')
        self.prior.load_state_dict(checkpoint['state_dict'])
        self.prior.eval()

        for param in self.prior.parameters():
            param.requires_grad = False

    def forward(self, x, s_last, mid_last):
        x0 = x
        out = self.F_B0(x0)
        mmp = self.prior(x0)

        out = self.MMAM1(out, mmp)

        out = self.F_B1(out)
        out = self.F_B2(out)

        mid = out

        out = torch.cat([out, mid_last, s_last], dim=1)

        out = self.F_R(out)
        s = self.F_h(out)

        return out, s, mid


# MMAM
class MMAMLayer(nn.Module):

    def __init__(self, para):
        super(MMAMLayer, self).__init__()
        self.n_feats = para['n_features']
        self.MMAM_conv0 = nn.Conv2d(1, self.n_feats // 2, 1)
        self.MMAM_conv1 = nn.Conv2d(self.n_feats // 2, self.n_feats, 1)

    def forward(self, x, y):
        scale = self.MMAM_conv1(
            F.leaky_relu(self.MMAM_conv0(y), 0.1, inplace=True))
        return x * (scale + 1)


# DownSampling module
class RDB_DS(nn.Module):

    def __init__(self,
                 in_channels,
                 growthRate,
                 out_channels,
                 num_layer,
                 activation='gelu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, out_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Dense layer
class dense_layer(nn.Module):

    def __init__(self, in_channels, growthRate, activation='gelu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Channel attention layer
class CALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y, y


class SALayer(nn.Module):

    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = conv1x1(1, 1)

    def forward(self, x, bm):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = avg_out
        y = torch.cat([y, bm], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y


# Residual dense block
class RDB(nn.Module):

    def __init__(self,
                 in_channels,
                 growthRate,
                 num_layer,
                 activation='gelu',
                 do_attention_ca=False,
                 do_attention_sa=False):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        self.do_attention_ca = do_attention_ca
        self.do_attention_sa = do_attention_sa
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)
        if do_attention_ca:
            self.ca = CALayer(in_channels_)
        if do_attention_sa:
            self.sa = SALayer()

    def forward(self, x):
        out = self.dense_layers(x)
        if self.do_attention_ca:
            out, _ = self.ca(out)
        if self.do_attention_sa:
            out_ = out
            out = self.sa(out_)
            out += out_

        out = self.conv1x1(out)
        out += x
        return out


# Middle network of residual dense blocks
class RDNet(nn.Module):

    def __init__(self,
                 in_channels,
                 growthRate,
                 num_layer,
                 num_blocks,
                 activation='gelu',
                 do_attention_ca=False,
                 do_attention_sa=False):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(
                RDB(in_channels,
                    growthRate,
                    num_layer,
                    activation,
                    do_attention_ca=do_attention_ca,
                    do_attention_sa=do_attention_sa))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


# Reconstructor
class Reconstructor(nn.Module):

    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para['future_frames']
        self.num_fb = para['past_frames']
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para['n_features']
        self.n_blocks = para['n_blocks_b']
        self.D_Net = RDNet(
            in_channels=(1 + 2 + 2) * self.n_feats,
            growthRate=2 * self.n_feats,
            num_layer=3,
            num_blocks=self.n_blocks,
            do_attention_ca=False,
            do_attention_sa=False)
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats),
                               2 * self.n_feats,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ConvTranspose2d(
                2 * self.n_feats,
                self.n_feats,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1), conv5x5(self.n_feats, 3, stride=1))

    def forward(self, x):
        x = self.D_Net(x)
        return self.model(x)


class CAT(nn.Module):

    def __init__(self, para):
        super(CAT, self).__init__()
        self.frames = para['future_frames'] + para['past_frames'] + 1
        self.n_feats = para['n_features']
        self.center = para['past_frames']
        self.fusion = conv1x1(self.frames * (5 * self.n_feats),
                              (5 * self.n_feats))
        self.ca = CALayer(self.frames * (5 * self.n_feats))

    def forward(self, hs):
        out = torch.cat(hs, dim=1)
        out, _ = self.ca(out)
        out = self.fusion(out)
        return out
