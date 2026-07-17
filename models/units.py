import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

# HD
class HDUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 1, 0, 0), 'd': (0, 1, 0, 1)}
    avg_factor = 2.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, [1, 2], stride=1, padding=0, dilation=1)
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x, conv1):
        K = 2
        S = 1
        # S = self.upscale 
        P = K - 1
        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)

        x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1]], dim=1)  # d

        x = x.unsqueeze(1).unsqueeze(1)
        x = conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B, -1, (H - P) * (W - P))
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        if self.ktype == "d":
            x = self.dconv_forward(x_in, self.conv1)
        else:  # "h"
            x = self.conv1(x_in)

        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)

        if self.ktype == "h":
            x = x.reshape(B, C, self.upscale * (H), self.upscale * (W - 1))
        elif self.ktype == "d":
            x = x.reshape(B, C, self.upscale * (H - 1), self.upscale * (W - 1))
        else:
            raise AttributeError

        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        if self.ktype == "h":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 1, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        elif self.ktype == "d":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 2, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        else:
            raise AttributeError
        return input_tensor

# HUnit
class HUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 0)}
    avg_factor = 4.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, [1,3], stride=1, padding=0, dilation=1)
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)

        if self.ktype=="h":
            x = self.conv1(x_in)

        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)

        if self.ktype == "h":
            x = x.reshape(B, C, self.upscale*(H), self.upscale*(W-2))
        else:
            raise AttributeError

        return torch.tanh(x)


    def get_lut_input(self, input_tensor):
        if self.ktype == "h":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 1, 3), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
            input_tensor = input_tensor_dil
        else:
            raise AttributeError
        return input_tensor

class GUnit(nn.Module):

    def __init__(self, nf=8, act=nn.ReLU):
        super(GUnit, self).__init__()
        self.act = act()
        self.conv1 = Conv(1, nf, 1)
        self.conv2 = Conv(nf, 3, 1)


    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B*C, 1, H, W)

        x = self.conv1(x_in)
        x = self.act(x)
        x = self.conv2(x)

        x = x.reshape(B, 3, H, W)
        return F.softmax(x, dim=1)#torch.tanh(x)


# PUnit : pivot
class PUnit(nn.Module):
    rot_dict = {'p': [0, 1, 2, 3]}
    pad_dict = {'p': (0, 0, 0, 0)}
    avg_factor = 1.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(PUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, 1, stride=1, padding=0, dilation=1)
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.conv1(x_in)
        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)

        x = x.reshape(B, C, self.upscale * (H), self.upscale * (W))

        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 1, 1), dtype=input_tensor.dtype)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
        input_tensor = input_tensor_dil

        return input_tensor


# HDV
class HDVUnit(nn.Module):
    rot_dict = {'h': [0, 2], 'd': [0, 1, 2, 3], 'v': [0, 2]}
    pad_dict = {'h': (0, 1, 0, 0), 'd': (0, 1, 0, 1), 'v': (0, 0, 0, 1)}
    avg_factor = 2.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDVUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        if self.ktype == "v":
            self.conv1 = Conv(1, nf, [2, 1], stride=1, padding=0, dilation=1)
        else:  # "h" or "d"
            self.conv1 = Conv(1, nf, [1, 2], stride=1, padding=0, dilation=1)

        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x, conv1):
        K = 2
        S = 1
        P = K - 1
        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)

        x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1]], dim=1)  # d

        x = x.unsqueeze(1).unsqueeze(1)
        x = conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B, -1, (H - P) * (W - P))
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        if self.ktype == "d":
            x = self.dconv_forward(x_in, self.conv1)
        else:  # "h" or "v"
            x = self.conv1(x_in)

        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.pixel_shuffle(x)

        if self.ktype == "h":
            x = x.reshape(B, C, self.upscale * (H), self.upscale * (W - 1))
        elif self.ktype == "d":
            x = x.reshape(B, C, self.upscale * (H - 1), self.upscale * (W - 1))
        elif self.ktype == "v":
            x = x.reshape(B, C, self.upscale * (H - 1), self.upscale * (W))
        else:
            raise AttributeError

        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        if self.ktype == "h":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 1, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        elif self.ktype == "d":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 2, 2), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        elif self.ktype == "v":
            input_tensor_dil = torch.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], 2, 1), dtype=input_tensor.dtype)
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 1]
            input_tensor = input_tensor_dil
        else:
            raise AttributeError
        return input_tensor


# HL
class HLUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'l': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'l': (0, 2, 0, 2)}
    avg_factor = 2.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HLUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'l':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'l':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor


# HS
class HSUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 's': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 's': (0, 2, 0, 2)}
    avg_factor = 2.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HSUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 's':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 's':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor


# HDB
class HDBUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2)}
    avg_factor = 3.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.flatten = nn.Flatten()

    def dconv_forward(self, x):
        K = 3
        S = 1
        # S = self.upscale
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        x = self.flatten(x)
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'd':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor


# HDBV
class HDBVUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'v': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2), 'v': (0, 2, 0, 2)}
    avg_factor = 4.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBVUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        elif self.ktype == 'v':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 0], x[:, :, 2, 0]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'd':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        elif self.ktype == 'v':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor


class HDBL2Unit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 6, 0, 6), 'd': (0, 6, 0, 6), 'b': (0, 6, 0, 6), 'l': (0, 6, 0, 6)}
    avg_factor = 4.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBL2Unit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 7
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 3], x[:, :, 0, 6]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 3, 3], x[:, :, 6, 6]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 3, 6], x[:, :, 6, 3]], dim=1)
        elif self.ktype == 'l':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 3, 0], x[:, :, 0, 3]], dim=1) ##change kernel shape
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 6), self.upscale * (W - 6))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 7, 7), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 3] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 6] = input_tensor[:, :, 2]
        elif self.ktype == 'd':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 3, 3] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 6, 6] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 3, 6] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 6, 3] = input_tensor[:, :, 2]
        elif self.ktype == 'l':  ##change kernel shape
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 3, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 3] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor


class HDBLUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2), 'l': (0, 2, 0, 2)}
    avg_factor = 4.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBLUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)


    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        elif self.ktype == 'l':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 0], x[:, :, 0, 1]], dim=1)  ##change kernel shape
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h':  # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'd':  # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        elif self.ktype == 'l':  ##change kernel shape
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor



class LUnit(nn.Module):
    avg_factor = 2.

    def __init__(self, nf=64, upscale=4, act=nn.ReLU):
        super(LUnit, self).__init__()
        self.upscale = upscale
        self.act = act()


        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 0]], dim=1)

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)

        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
        input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
        input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 2]

        input_tensor = input_tensor_dil
        return input_tensor

class HDBLRC1Unit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3], 'r': [0, 1, 2, 3], 'c': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 4, 0, 4), 'd': (0, 4, 0, 4), 'b': (0, 4, 0, 4), 'l': (0, 4, 0, 4), 'r': (0, 4, 0, 4), 'c': (0, 4, 0, 4)}
    avg_factor = 6.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBLRC1Unit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 5
        S = 1
        # S = self.upscale
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 2], x[:, :, 0, 4]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 2], x[:, :, 4, 4]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 4], x[:, :, 4, 2]], dim=1)
        elif self.ktype == 'l':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 2, 0], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'r':
            x = torch.cat([x[:, :, 2, 4], x[:, :, 4, 4], x[:, :, 4, 2]], dim=1)
        elif self.ktype == 'c':
            x = torch.cat([x[:, :, 2, 0], x[:, :, 4, 0], x[:, :, 4, 2]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)
        #x_in = x_in * 2 - 1

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 4), self.upscale * (W - 4))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 5, 5), dtype=input_tensor.dtype)
        if self.ktype == 'h': # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 4] = input_tensor[:, :, 2]
        elif self.ktype == 'd': # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 4, 4] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 2, 4] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 4, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'l': # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'r': # red
            input_tensor_dil[:, :, 2, 4] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 4, 4] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 4, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'c':
            input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 4, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 4, 2] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor

class HDBLRCUnit(nn.Module):
    rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3], 'r': [0, 1, 2, 3], 'c': [0, 1, 2, 3]}
    pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2), 'l': (0, 2, 0, 2), 'r': (0, 2, 0, 2), 'c': (0, 2, 0, 2)}
    avg_factor = 6.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(HDBLRCUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()


        self.conv1 = Conv(1, nf, (1, 3))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        K = 3
        S = 1
        # S = self.upscale
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 'h':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 0, 2]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'b':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        elif self.ktype == 'l':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 0], x[:, :, 0, 1]], dim=1)
        elif self.ktype == 'r':
            x = torch.cat([x[:, :, 1, 2], x[:, :, 2, 2], x[:, :, 2, 1]], dim=1)
        elif self.ktype == 'c':
            x = torch.cat([x[:, :, 1, 0], x[:, :, 2, 0], x[:, :, 2, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)

    def get_lut_input(self, input_tensor):
        input_tensor_dil = torch.zeros((input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype)
        if self.ktype == 'h': # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'd': # red
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 2]
        elif self.ktype == 'b':
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        elif self.ktype == 'l': # green
            input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 0, 1] = input_tensor[:, :, 2]
        elif self.ktype == 'r': # red
            input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        elif self.ktype == 'c':
            input_tensor_dil[:, :, 1, 0] = input_tensor[:, :, 0]
            input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1]
            input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 2]
        else:
            raise AttributeError

        input_tensor = input_tensor_dil
        return input_tensor

############### MuLUT Blocks ###############
class MuLUTUnit(nn.Module):
    """ Generalized (spatial-wise)  MuLUT block. """

    def __init__(self, mode, nf, upscale=1, out_c=1, dense=True):
        super(MuLUTUnit, self).__init__()
        self.act = nn.ReLU()
        self.upscale = upscale

        if mode == '2x2':
            self.conv1 = Conv(1, nf, 2)
        elif mode == '2x2d':
            self.conv1 = Conv(1, nf, 2, dilation=2)
        elif mode == '2x2d3':
            self.conv1 = Conv(1, nf, 2, dilation=3)
        elif mode == '1x4':
            self.conv1 = Conv(1, nf, (1, 4))
        else:
            raise AttributeError

        if dense:
            self.conv2 = DenseConv(nf, nf)
            self.conv3 = DenseConv(nf + nf * 1, nf)
            self.conv4 = DenseConv(nf + nf * 2, nf)
            self.conv5 = DenseConv(nf + nf * 3, nf)
            self.conv6 = Conv(nf * 5, 1 * upscale * upscale, 1)
        else:
            self.conv2 = ActConv(nf, nf, 1)
            self.conv3 = ActConv(nf, nf, 1)
            self.conv4 = ActConv(nf, nf, 1)
            self.conv5 = ActConv(nf, nf, 1)
            self.conv6 = Conv(nf, upscale * upscale, 1)
        if self.upscale > 1:
            self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.tanh(self.conv6(x))
        if self.upscale > 1:
            x = self.pixel_shuffle(x)
        return x


# MuLUT-SDY
class SDYUnit(nn.Module):
    rot_dict = {'s': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'y': [0, 1, 2, 3]}
    pad_dict = {'s': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'y': (0, 2, 0, 2)}
    avg_factor = 4.

    def __init__(self, ktype, nf=64, upscale=4, act=nn.ReLU):
        super(SDYUnit, self).__init__()
        self.ktype = ktype
        self.upscale = upscale
        self.act = act()

        self.conv1 = Conv(1, nf, (1, 4))
        self.conv2 = ActConv(nf, nf, 1, act=act)
        self.conv3 = ActConv(nf, nf, 1, act=act)
        self.conv4 = ActConv(nf, nf, 1, act=act)
        self.conv5 = ActConv(nf, nf, 1, act=act)
        self.conv6 = Conv(nf, upscale * upscale, 1)

        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def dconv_forward(self, x):
        # if self.ktype == 's':
        #     K = 2
        # else:
        K = 3
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, (H - P) * (W - P))
        x = x.permute((0, 1, 3, 2))
        x = x.reshape(B * C * (H - P) * (W - P), K, K)
        x = x.unsqueeze(1)
        if self.ktype == 's':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 0], x[:, :, 1, 1]], dim=1)
        elif self.ktype == 'd':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 2], x[:, :, 2, 0], x[:, :, 2, 2]], dim=1)
        elif self.ktype == 'y':
            x = torch.cat([x[:, :, 0, 0], x[:, :, 1, 1], x[:, :, 1, 2], x[:, :, 2, 1]], dim=1)
        # elif self.ktype == 'l':
        #     x = torch.cat([x[:, :, 0, 0], x[:, :, 0, 1], x[:, :, 1, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        x = x.reshape(B, C, (H - P) * (W - P), -1)  # B,C,K*K,L
        x = x.permute((0, 1, 3, 2))  # B,C,K*K,L
        x = x.reshape(B, -1, (H - P) * (W - P))  # B,C*K*K,L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, C, Hout, Wout

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x_in = x_in.reshape(B * C, 1, H, W)

        x = self.act(self.dconv_forward(x_in))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(B, C, self.upscale * (H - 2), self.upscale * (W - 2))
        return torch.tanh(x)



class PCM_comb_Module_4in(nn.Module):
    """PCM module for YU channel with Dense Connections"""
    rot_dict = {'a': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'c': [0, 1, 2, 3]}
    pad_dict = {'a': (0, 1, 0, 1), 'b': (0, 1, 0, 1), 'c': (0, 1, 0, 1)}
    avg_factor = 3.

    def __init__(self, ktype, nf=64):
        super(PCM_comb_Module_4in, self).__init__()
        self.ktype = ktype
        # 第一层：YU通道对的1x2卷积
        self.conv1 = nn.Conv2d(1, nf, kernel_size=(1, 4), stride=1, padding=0)
        self.act = nn.GELU()

        # 密集连接层（模仿DnLUT）
        self.conv2 = self._make_dense_conv(nf, nf)  # 输入nf，输出nf+nf
        self.conv3 = self._make_dense_conv(nf + nf, nf)  # 输入nf*2，输出nf*3
        self.conv4 = self._make_dense_conv(nf * 3, nf)  # 输入nf*3，输出nf*4
        self.conv5 = self._make_dense_conv(nf * 4, nf)  # 输入nf*4，输出nf*5

        # 最终输出层
        self.conv6 = nn.Conv2d(nf * 5, 1, 1)  # 输入nf*5，输出1通道


        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def dconv_forward(self, x):
        K = 2
        S = 1
        P = K - 1

        B, C, H, W = x.shape
        L = (H - P) * (W - P)
        x = F.unfold(x, K)
        x = x.view(B, C, K * K, L)
        x = x.permute((0, 3, 1, 2))  # [B, L, C, K*K]
        x = x.reshape(B * L, C, K, K)
        if self.ktype == 'a':
            x = torch.cat([x[:, 0:1, 0, 0], x[:, 0:1, 0, 1], x[:, 1:2, 0, 0], x[:, 1:2, 0, 1]], dim=1)  # [B*L, 3]
        elif self.ktype == 'b':
            x = torch.cat([x[:, 1:2, 0, 0], x[:, 1:2, 0, 1], x[:, 2:, 0, 0], x[:, 2:, 0, 1]], dim=1)
        elif self.ktype == 'c':
            x = torch.cat([x[:, 2:, 0, 0], x[:, 2:, 0, 1], x[:, 0:1, 0, 0], x[:, 0:1, 0, 1]], dim=1)
        else:
            raise AttributeError

        x = x.unsqueeze(1).unsqueeze(1)  # [B*L, 1, 1, 3]
        x = self.conv1(x)  # [B*L, nf, 1, 1]
        x = x.squeeze(1)  # [B*L, nf, 1]
        x = x.reshape(B, L, -1)  # B,L,nf
        x = x.permute((0, 2, 1))  # B, nf, L
        return F.fold(x, ((H - P) * S, (W - P) * S), S, stride=S)  # B, nf, Hout, Wout

    def _make_dense_conv(self, in_channels, out_channels):
        """创建密集连接层"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU()
        )

    def forward(self, x_in):
        """
        yu_input: Y通道输入 [B, 2, H, W+1]
        返回: 增强后的Y通道 [B, 1, H, W]
        """
        B, C, H, W = x_in.shape  # [B, 2, H, W+1]
        x_in = x_in * 2 - 1

        # PCM处理（模仿DnLUT的密集连接结构）
        x = self.act(self.dconv_forward(x_in))  # [B, nf, H, W]

        # 密集连接处理
        feat1 = self.conv2(x)  # [B, nf, H, W]
        x = torch.cat([x, feat1], dim=1)  # [B, nf*2, H, W]

        feat2 = self.conv3(x)  # [B, nf, H, W]
        x = torch.cat([x, feat2], dim=1)  # [B, nf*3, H, W]

        feat3 = self.conv4(x)  # [B, nf, H, W]
        x = torch.cat([x, feat3], dim=1)  # [B, nf*4, H, W]

        feat4 = self.conv5(x)  # [B, nf, H, W]
        x = torch.cat([x, feat4], dim=1)  # [B, nf*5, H, W]

        pcm_out = torch.tanh(self.conv6(x))

        return pcm_out

    def get_lut_input(self, input_tensor):
        """
        input_tensor: [N, 1, 4]
        输出: [N, 2, 1, 2] —— 和 forward 的 dconv_forward 完全一致
        """
        N = input_tensor.shape[0]
        input_tensor_dil = torch.zeros((N, 3, 2, 2), dtype=input_tensor.dtype)#(2,2)

        if self.ktype == 'a':
            # YU
            input_tensor_dil[:, 0, 0, 0] = input_tensor[:, 0, 0, 0]
            input_tensor_dil[:, 0, 0, 1] = input_tensor[:, 0, 0, 1]
            input_tensor_dil[:, 1, 0, 0] = input_tensor[:, 0, 0, 2]
            input_tensor_dil[:, 1, 0, 1] = input_tensor[:, 0, 0, 3]

        elif self.ktype == 'b':
            # UV
            input_tensor_dil[:, 1, 0, 0] = input_tensor[:, 0, 0, 0]
            input_tensor_dil[:, 1, 0, 1] = input_tensor[:, 0, 0, 1]
            input_tensor_dil[:, 2, 0, 0] = input_tensor[:, 0, 0, 2]
            input_tensor_dil[:, 2, 0, 1] = input_tensor[:, 0, 0, 3]

        elif self.ktype == 'c':
            # VY
            input_tensor_dil[:, 2, 0, 0] = input_tensor[:, 0, 0, 0]
            input_tensor_dil[:, 2, 0, 1] = input_tensor[:, 0, 0, 1]
            input_tensor_dil[:, 0, 0, 0] = input_tensor[:, 0, 0, 2]
            input_tensor_dil[:, 0, 0, 1] = input_tensor[:, 0, 0, 3]

        else:
            raise AttributeError
        input_tensor = input_tensor_dil
        return input_tensor
