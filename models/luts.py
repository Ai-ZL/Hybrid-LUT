import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import bit_plane_slicing, decode_bit_mask, round_func

class HDVLUT(nn.Module):
    def __init__(self, h_weight, d_weight, v_weight, L, upscale=2):
        super(HDVLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.v_weight = v_weight
        self.rot_dict = {'h': [0, 2], 'd': [0, 1, 2, 3], 'v': [0, 2]}
        self.pad_dict = {'h': (0,1,0,0), 'd': (0,1,0,1), 'v': (0,0,0,1)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'v']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 0:0+H, 1:1+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 1:1+H, 1:1+W]
                else: # v
                    img_a = img_in[:,:, 0:0+H, 0:0+W]
                    img_b = img_in[:,:, 1:1+H, 0:0+W]
                    weight = self.v_weight

                tmp = weight[img_a.flatten()*self.L + img_b.flatten()].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute(0, 1, 2, 4, 3, 5).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HLLUT(nn.Module):
    def __init__(self, h_weight, l_weight, L, upscale=2):
        super(HLLUT, self).__init__()
        self.h_weight = h_weight
        self.l_weight = l_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'l': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 'l': (0, 2, 0, 2)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'l']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 'l':
                    weight = self.l_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 1:1+H, 1:1+W]

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HSLUT(nn.Module):
    def __init__(self, h_weight, s_weight, L, upscale=2):
        super(HSLUT, self).__init__()
        self.h_weight = h_weight
        self.s_weight = s_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 's': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 's': (0, 2, 0, 2)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 's']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 's':
                    weight = self.s_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HSLUTft(nn.Module):
    def __init__(self, h_weight, s_weight, L, upscale=2):
        super(HSLUTft, self).__init__()
        self.h_weight = nn.Parameter(
            torch.as_tensor(h_weight, dtype=torch.float32)
        )
        self.s_weight = nn.Parameter(
            torch.as_tensor(s_weight, dtype=torch.float32)
        )
        self.rot_dict = {'h': [0, 1, 2, 3], 's': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 's': (0, 2, 0, 2)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 's']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 's':
                    weight = self.s_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor

class HDBL2LUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, l_weight, L, upscale=2):
        super(HDBT2LUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.l_weight = l_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 6, 0, 6)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'l']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 3:3 + W]
                    img_c = img_in[:, :, 0:0 + H, 6:6 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 3:3 + W]
                    img_c = img_in[:, :, 6:6 + H, 6:6 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 6:6 + W]
                    img_c = img_in[:, :, 6:6 + H, 3:3 + W]

                elif ktype == 'l':
                    weight = self.l_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 0:0 + W]
                    img_c = img_in[:, :, 0:0 + H, 3:3 + W]

                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        final_out = out / self.avg_factor
        return final_out

class HDBL2LUTft(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, t_weight, L, upscale=2):
        super(HDBL2LUTft, self).__init__()
        self.h_weight = nn.Parameter(
            torch.as_tensor(h_weight, dtype=torch.float32)
        )
        self.d_weight = nn.Parameter(
            torch.as_tensor(d_weight, dtype=torch.float32)
        )
        self.b_weight = nn.Parameter(
            torch.as_tensor(b_weight, dtype=torch.float32)
        )
        self.l_weight = nn.Parameter(
            torch.as_tensor(l_weight, dtype=torch.float32)
        )
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 6, 0, 6)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'l']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 3:3 + W]
                    img_c = img_in[:, :, 0:0 + H, 6:6 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 3:3 + W]
                    img_c = img_in[:, :, 6:6 + H, 6:6 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 6:6 + W]
                    img_c = img_in[:, :, 6:6 + H, 3:3 + W]

                elif ktype == 'l':
                    weight = self.l_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 3:3 + H, 0:0 + W]
                    img_c = img_in[:, :, 0:0 + H, 3:3 + W]

                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        final_out = out / self.avg_factor
        return final_out


class HDBLRC1LUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, l_weight, r_weight, c_weight, L, upscale=2):
        super(HDBLRC1LUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.l_weight = l_weight
        self.r_weight = r_weight
        self.c_weight = c_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3], 'r': [0, 1, 2, 3], 'c': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 4, 0, 4), 'd': (0, 4, 0, 4), 'b': (0, 4, 0, 4), 'l': (0, 4, 0, 4), 'r': (0, 4, 0, 4), 'c': (0, 4, 0, 4)}
        self.avg_factor = 6.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'l', 'r', 'c']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 2:2 + W]
                    img_c = img_in[:, :, 0:0 + H, 4:4 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 2:2 + W]
                    img_c = img_in[:, :, 4:4 + H, 4:4 + W]
                elif ktype == 'b':
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 4:4 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]
                    weight = self.b_weight
                elif ktype == 'l':
                    weight = self.l_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 0:0 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'r':
                    weight = self.r_weight
                    img_a = img_in[:, :, 2:2 + H, 4:4 + W]
                    img_b = img_in[:, :, 4:4 + H, 4:4 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]
                elif ktype == 'c':
                    img_a = img_in[:, :, 2:2 + H, 0:0 + W]
                    img_b = img_in[:, :, 4:4 + H, 0:0 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]
                    weight = self.c_weight

                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor


class HDBLRC1LUTft(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, l_weight, r_weight, c_weight, L, upscale=2):
        super(HDBLRC1LUTft, self).__init__()
        self.h_weight = nn.Parameter(
            torch.as_tensor(h_weight, dtype=torch.float32)
        )
        self.d_weight = nn.Parameter(
            torch.as_tensor(d_weight, dtype=torch.float32)
        )
        self.b_weight = nn.Parameter(
            torch.as_tensor(b_weight, dtype=torch.float32)
        )
        self.l_weight = nn.Parameter(
            torch.as_tensor(l_weight, dtype=torch.float32)
        )
        self.r_weight = nn.Parameter(
            torch.as_tensor(r_weight, dtype=torch.float32)
        )
        self.c_weight = nn.Parameter(
            torch.as_tensor(c_weight, dtype=torch.float32)
        )
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3], 'l': [0, 1, 2, 3], 'r': [0, 1, 2, 3], 'c': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 4, 0, 4), 'd': (0, 4, 0, 4), 'b': (0, 4, 0, 4), 'l': (0, 4, 0, 4), 'r': (0, 4, 0, 4), 'c': (0, 4, 0, 4)}
        self.avg_factor = 6.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'l', 'r', 'c']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 2:2 + W]
                    img_c = img_in[:, :, 0:0 + H, 4:4 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 2:2 + W]
                    img_c = img_in[:, :, 4:4 + H, 4:4 + W]
                elif ktype == 'b':
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 4:4 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]

                    weight = self.b_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)
                elif ktype == 'l':
                    weight = self.l_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 2:2 + H, 0:0 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'r':
                    weight = self.r_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a = img_in[:, :, 2:2 + H, 4:4 + W]
                    img_b = img_in[:, :, 4:4 + H, 4:4 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]
                elif ktype == 'c':
                    img_a = img_in[:, :, 2:2 + H, 0:0 + W]
                    img_b = img_in[:, :, 4:4 + H, 0:0 + W]
                    img_c = img_in[:, :, 4:4 + H, 2:2 + W]
                    weight = self.c_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor


class LLUT(nn.Module):
    def __init__(self, l_weight, L, upscale=2):
        super(LLUT, self).__init__()
        self.l_weight = l_weight
        self.rot_dict = [0,2]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for r in self.rot_dict:
            img_lr_rot = torch.rot90(img_lr, r, [2, 3])
            _, _, H, W = img_lr_rot.shape
            img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)

            weight = self.l_weight
            img_a = img_in[:, :, 0:0 + H, 0:0 + W]
            img_b = img_in[:, :, 0:0 + H, 1:1 + W]
            img_c = img_in[:, :, 1:1 + H, 0:0 + W]

            tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                         ].reshape(
                (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
            tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
            out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor


class LLUTft(nn.Module):
    def __init__(self, l_weight, L, upscale=2):
        super(LLUTft, self).__init__()
        self.l_weight = nn.Parameter(
            torch.as_tensor(l_weight, dtype=torch.float32)
        )
        self.rot_dict = [0,2]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for r in self.rot_dict:
            img_lr_rot = torch.rot90(img_lr, r, [2, 3])
            _, _, H, W = img_lr_rot.shape
            img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)

            weight = self.l_weight
            weight = weight * 127
            weight = round_func(weight)
            weight = torch.clamp(weight, -127, 127)
            
            img_a = img_in[:, :, 0:0 + H, 0:0 + W]
            img_b = img_in[:, :, 0:0 + H, 1:1 + W]
            img_c = img_in[:, :, 1:1 + H, 0:0 + W]

            tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                         ].reshape(
                (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
            tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
            out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor


class HDBLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, L, upscale=2):
        super(HDBLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3], 'b': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 2, 0, 2), 'd': (0, 2, 0, 2), 'b': (0, 2, 0, 2)}
        self.avg_factor = 3.

        self.L = L
        self.upscale = upscale
        
    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2,3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 0:0+H, 1:1+W]
                    img_c = img_in[:, :, 0:0+H, 2:2+W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 1:1+W]
                    img_c = img_in[:, :, 2:2+H, 2:2+W]
                else:
                    img_a = img_in[:, :, 0:0+H, 0:0+W]
                    img_b = img_in[:, :, 1:1+H, 2:2+W]
                    img_c = img_in[:, :, 2:2+H, 1:1+W]
                    weight = self.b_weight

                tmp = weight[img_a.flatten()*self.L*self.L + img_b.flatten()*self.L + img_c.flatten()
                             ].reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))   
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape((img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2,3])

        return out/self.avg_factor


class HDLUT(nn.Module):
    def __init__(self, h_weight, d_weight, L, upscale=2):
        super(HDLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.rot_dict = {'h': [0, 1, 2, 3], 'd': [0, 1, 2, 3]}
        self.pad_dict = {'h': (0, 1, 0, 0), 'd': (0, 1, 0, 1)}
        self.avg_factor = 2.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd']:
            for r in self.rot_dict[ktype]:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict[ktype], mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                else:
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]


                tmp = weight[img_a.flatten() * self.L + img_b.flatten()].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                out += torch.rot90(tmp, 4 - r, [2, 3])

        return out / self.avg_factor

class HDBVLUT(nn.Module):
    def __init__(self, h_weight, d_weight, b_weight, v_weight, L, upscale=2):
        super(HDBVLUT, self).__init__()
        self.h_weight = h_weight
        self.d_weight = d_weight
        self.b_weight = b_weight
        self.v_weight = v_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 2, 0, 2)
        self.avg_factor = 4.

        self.L = L
        self.upscale = upscale

    def forward(self, img_lr):
        out = 0.

        for ktype in ['h', 'd', 'b', 'v']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_lr, r, [2, 3])
                _, _, H, W = img_lr_rot.shape
                img_in = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                if ktype == 'h':
                    weight = self.h_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 0:0 + H, 1:1 + W]
                    img_c = img_in[:, :, 0:0 + H, 2:2 + W]
                elif ktype == 'd':
                    weight = self.d_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 1:1 + W]
                    img_c = img_in[:, :, 2:2 + H, 2:2 + W]
                elif ktype == 'b':
                    weight = self.b_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 2:2 + W]
                    img_c = img_in[:, :, 2:2 + H, 1:1 + W]

                elif ktype == 'v':
                    weight = self.v_weight
                    img_a = img_in[:, :, 0:0 + H, 0:0 + W]
                    img_b = img_in[:, :, 1:1 + H, 0:0 + W]
                    img_c = img_in[:, :, 2:2 + H, 0:0 + W]


                tmp = weight[img_a.flatten() * self.L * self.L + img_b.flatten() * self.L + img_c.flatten()
                             ].reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2], img_a.shape[3], self.upscale, self.upscale))
                tmp = tmp.permute((0, 1, 2, 4, 3, 5)).reshape(
                    (img_a.shape[0], img_a.shape[1], img_a.shape[2] * self.upscale, img_a.shape[3] * self.upscale))
                tmp = torch.rot90(tmp, 4 - r, [2, 3])
                out += tmp

        return out / self.avg_factor


class PCMComb4inLUT(nn.Module):
    def __init__(self, a_weight, b_weight, c_weight, interval=4):
        """
        PCM LUT推理模块
        lut_path: PCM LUT文件路径
        interval: 采样间隔，与生成LUT时一致
        """
        super(PCMComb4inLUT, self).__init__()
        self.a_weight = a_weight
        self.b_weight = b_weight
        self.c_weight = c_weight
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 1, 0, 1)
        self.avg_factor = 3.
        self.interval = interval
        self.L = 2 ** (8 - interval) + 1  # 每个维度的采样点数

    def forward(self, img_in):
        """
        三维单纯形插值 - 适配PCM的YU输入格式
        img_in: [B, 3, H, W] 的YU输入
        """

        out_all = 0.
        device = img_in.device

        for ktype in ['a', 'b', 'c']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_in, r, [2, 3])
                _, C, H, W = img_lr_rot.shape
                q = 2 ** self.interval
                upscale = 1
                img_lr_pad = F.pad(img_lr_rot, self.pad_dict, mode='replicate').type(torch.int64)
                img_lr_np = img_lr_pad.cpu().numpy().astype(np.float32)
                img_lr_np = img_lr_np[0, :, :, :]
                if ktype == 'a':
                    weight = self.a_weight.numpy()
                    img_a1 = img_lr_np[0:1, 0:0 + H, 0:0 + W] // q
                    img_b1 = img_lr_np[0:1, 0:0 + H, 1:1 + W] // q
                    img_c1 = img_lr_np[1:2, 0:0 + H, 0:0 + W] // q
                    img_d1 = img_lr_np[1:2, 0:0 + H, 1:1 + W] // q

                    fa = img_lr_np[0:1, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[0:1, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[1:2, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[1:2, 0:0 + H, 1:1 + W] % q

                elif ktype == 'b':
                    weight = self.b_weight.numpy()
                    img_a1 = img_lr_np[1:2, 0:0 + H, 0:0 + W] // q
                    img_b1 = img_lr_np[1:2, 0:0 + H, 1:1 + W] // q
                    img_c1 = img_lr_np[2:, 0:0 + H, 0:0 + W] // q
                    img_d1 = img_lr_np[2:, 0:0 + H, 1:1 + W] // q

                    fa = img_lr_np[1:2, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[1:2, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[2:, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[2:, 0:0 + H, 1:1 + W] % q
                elif ktype == 'c':
                    weight = self.c_weight.numpy()
                    img_a1 = img_lr_np[2:, 0:0 + H, 0:0 + W] // q
                    img_b1 = img_lr_np[2:, 0:0 + H, 1:1 + W] // q
                    img_c1 = img_lr_np[0:1, 0:0 + H, 0:0 + W] // q
                    img_d1 = img_lr_np[0:1, 0:0 + H, 1:1 + W] // q

                    fa = img_lr_np[2:, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[2:, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[0:1, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[0:1, 0:0 + H, 1:1 + W] % q


                img_a2 = img_a1 + 1
                img_b2 = img_b1 + 1
                img_c2 = img_c1 + 1
                img_d2 = img_d1 + 1

                # 计算16个顶点的LUT值
                L = self.L
                p0000 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0001 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0010 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0011 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0100 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0101 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0110 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p0111 = weight[img_a1.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1000 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1001 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1010 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1011 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b1.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1100 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1101 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c1.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1110 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d1.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                p1111 = weight[img_a2.flatten().astype(np.int_) * L * L * L + img_b2.flatten().astype(
                    np.int_) * L * L + img_c2.flatten().astype(np.int_) * L + img_d2.flatten().astype(np.int_)].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))

                out = np.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2]
                out = out.reshape(sz, -1).astype(np.float32)

                p0000 = p0000.reshape(sz, -1).astype(np.float32)

                p0100 = p0100.reshape(sz, -1).astype(np.float32)
                p1000 = p1000.reshape(sz, -1).astype(np.float32)
                p1100 = p1100.reshape(sz, -1).astype(np.float32)
                fa = fa.reshape(-1, 1).astype(np.float32)

                p0001 = p0001.reshape(sz, -1).astype(np.float32)
                p0101 = p0101.reshape(sz, -1).astype(np.float32)
                p1001 = p1001.reshape(sz, -1).astype(np.float32)
                p1101 = p1101.reshape(sz, -1).astype(np.float32)
                fb = fb.reshape(-1, 1).astype(np.float32)
                fc = fc.reshape(-1, 1).astype(np.float32)

                p0010 = p0010.reshape(sz, -1).astype(np.float32)
                p0110 = p0110.reshape(sz, -1).astype(np.float32)
                p1010 = p1010.reshape(sz, -1).astype(np.float32)
                p1110 = p1110.reshape(sz, -1).astype(np.float32)
                fd = fd.reshape(-1, 1).astype(np.float32)

                p0011 = p0011.reshape(sz, -1).astype(np.float32)
                p0111 = p0111.reshape(sz, -1).astype(np.float32)
                p1011 = p1011.reshape(sz, -1).astype(np.float32)
                p1111 = p1111.reshape(sz, -1).astype(np.float32)

                fab = fa > fb;
                fac = fa > fc;
                fad = fa > fd

                fbc = fb > fc;
                fbd = fb > fd;
                fcd = fc > fd

                i1 = i = np.logical_and.reduce((fab, fbc, fcd)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (
                            fc[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]
                i2 = i = np.logical_and.reduce((~i1[:, None], fab, fbc, fbd)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (
                            fd[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]
                i3 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], fab, fbc, fad)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (
                            fb[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]
                i4 = i = np.logical_and.reduce((~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc)).squeeze(1)

                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (
                            fb[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]

                i5 = i = np.logical_and.reduce((~(fbc), fab, fac, fbd)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (
                            fb[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]
                i6 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], fab, fac, fcd)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (
                            fd[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]
                i7 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad)).squeeze(1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (
                            fc[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]
                i8 = i = np.logical_and.reduce((~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac)).squeeze(1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (
                            fc[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]

                i9 = i = np.logical_and.reduce((~(fbc), ~(fac), fab, fbd)).squeeze(1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (
                            fb[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]

                i10 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], fab, fad)).squeeze(1)  # c > a > d > b
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (
                            fd[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]
                i11 = i = np.logical_and.reduce((~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd)).squeeze(
                    1)  # c > d > a > b
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (
                            fa[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]
                i12 = i = np.logical_and.reduce(
                    (~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab)).squeeze(1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (
                            fa[i] - fb[i]) * \
                         p1011[
                             i] + (fb[i]) * p1111[i]

                i13 = i = np.logical_and.reduce((~(fab), fac, fcd)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (
                            fc[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]
                i14 = i = np.logical_and.reduce((~(fab), ~i13[:, None], fac, fad)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (
                            fd[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]
                i15 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], fac, fbd)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (
                            fa[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]
                i16 = i = np.logical_and.reduce((~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac)).squeeze(1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (
                            fa[i] - fc[i]) * \
                         p1101[
                             i] + (fc[i]) * p1111[i]

                i17 = i = np.logical_and.reduce((~(fab), ~(fac), fbc, fad)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (
                            fa[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]
                i18 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], fbc, fcd)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (
                            fd[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]
                i19 = i = np.logical_and.reduce((~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd)).squeeze(1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (
                            fc[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]
                i20 = i = np.logical_and.reduce(
                    (~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc)).squeeze(1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (
                            fc[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]

                i21 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), fad)).squeeze(1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (
                            fa[i] - fd[i]) * \
                         p1110[
                             i] + (fd[i]) * p1111[i]
                i22 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd)).squeeze(1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (
                            fd[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]
                i23 = i = np.logical_and.reduce((~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd)).squeeze(1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (
                            fb[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]
                i24 = i = np.logical_and.reduce(
                    (~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None])).squeeze(
                    1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (
                            fb[i] - fa[i]) * \
                         p0111[
                             i] + (fa[i]) * p1111[i]


                out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], upscale, upscale))
                out = np.transpose(out, (0, 1, 3, 2, 4)).reshape(
                    (img_a1.shape[0], img_a1.shape[1] * upscale, img_a1.shape[2] * upscale))

                # 归一化到[-1,1]范围
                # out = out.astype(np.float32) / q / 127.0
                out = out / q
                out = torch.from_numpy(out).to(device)
                out = out[np.newaxis, :, :, :]
                out = torch.rot90(out, 4 - r, [2, 3])

                out_all += out


        final_out = out_all / self.avg_factor
        return final_out


class PCMComb4inLUTft(nn.Module):
    def __init__(self, a_weight, b_weight, c_weight, interval=4):
        """
        PCM LUT推理模块
        lut_path: PCM LUT文件路径
        interval: 采样间隔，与生成LUT时一致
        """
        super(PCMComb4inLUTft, self).__init__()
        self.a_weight = nn.Parameter(
            torch.as_tensor(a_weight, dtype=torch.float32)
        )
        self.b_weight = nn.Parameter(
            torch.as_tensor(b_weight, dtype=torch.float32)
        )
        self.c_weight = nn.Parameter(
            torch.as_tensor(c_weight, dtype=torch.float32)
        )
        self.rot_dict = [0, 1, 2, 3]
        self.pad_dict = (0, 1, 0, 1)
        self.avg_factor = 3.
        self.interval = interval
        self.L = 2 ** (8 - interval) + 1  # 每个维度的采样点数

    def forward(self, img_in):
        """
        三维单纯形插值 - 适配PCM的YU输入格式
        img_in: [B, 3, H, W] 的YU输入
        """

        out_all = 0.
        device = img_in.device

        for ktype in ['a', 'b', 'c']:
            for r in self.rot_dict:
                img_lr_rot = torch.rot90(img_in, r, [2, 3])
                _, C, H, W = img_lr_rot.shape
                q = 2 ** self.interval
                upscale = 1
                img_lr_pad = F.pad(img_lr_rot, self.pad_dict, mode='replicate')#.type(torch.int64)
                img_lr_np = img_lr_pad#[0, :, :, :].type(torch.float32)
                if ktype == 'a':
                    weight = self.a_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a1 = torch.floor_divide(img_lr_np[:, 0:1, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_b1 = torch.floor_divide(img_lr_np[:, 0:1, 0:0 + H, 1:1 + W], q).type(torch.int64)
                    img_c1 = torch.floor_divide(img_lr_np[:, 1:2, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_d1 = torch.floor_divide(img_lr_np[:, 1:2, 0:0 + H, 1:1 + W], q).type(torch.int64)

                    fa = img_lr_np[:, 0:1, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[:, 0:1, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[:, 1:2, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[:, 1:2, 0:0 + H, 1:1 + W] % q

                elif ktype == 'b':
                    weight = self.b_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a1 = torch.floor_divide(img_lr_np[:, 1:2, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_b1 = torch.floor_divide(img_lr_np[:, 1:2, 0:0 + H, 1:1 + W], q).type(torch.int64)
                    img_c1 = torch.floor_divide(img_lr_np[:, 2:, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_d1 = torch.floor_divide(img_lr_np[:, 2:, 0:0 + H, 1:1 + W], q).type(torch.int64)

                    fa = img_lr_np[:, 1:2, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[:, 1:2, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[:, 2:, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[:, 2:, 0:0 + H, 1:1 + W] % q
                elif ktype == 'c':
                    weight = self.c_weight
                    weight = weight * 127
                    weight = round_func(weight)
                    weight = torch.clamp(weight, -127, 127)

                    img_a1 = torch.floor_divide(img_lr_np[:, 2:, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_b1 = torch.floor_divide(img_lr_np[:, 2:, 0:0 + H, 1:1 + W], q).type(torch.int64)
                    img_c1 = torch.floor_divide(img_lr_np[:, 0:1, 0:0 + H, 0:0 + W], q).type(torch.int64)
                    img_d1 = torch.floor_divide(img_lr_np[:, 0:1, 0:0 + H, 1:1 + W], q).type(torch.int64)

                    fa = img_lr_np[:, 2:, 0:0 + H, 0:0 + W] % q
                    fb = img_lr_np[:, 2:, 0:0 + H, 1:1 + W] % q
                    fc = img_lr_np[:, 0:1, 0:0 + H, 0:0 + W] % q
                    fd = img_lr_np[:, 0:1, 0:0 + H, 1:1 + W] % q


                img_a2 = img_a1 + 1
                img_b2 = img_b1 + 1
                img_c2 = img_c1 + 1
                img_d2 = img_d1 + 1

                # 计算16个顶点的LUT值
                L = self.L
                p0000 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0001 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0010 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0011 = weight[img_a1.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0100 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0101 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0110 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p0111 = weight[img_a1.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1000 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1001 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1010 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1011 = weight[img_a2.flatten() * L * L * L + img_b1.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1100 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1101 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c1.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1110 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d1.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                p1111 = weight[img_a2.flatten() * L * L * L + img_b2.flatten() * L * L + img_c2.flatten() * L + img_d2.flatten()].reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))

                out = torch.zeros((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale), dtype=weight.dtype).to(device=weight.device)
                sz = img_a1.shape[0] * img_a1.shape[1] * img_a1.shape[2] * img_a1.shape[3]
                out = out.reshape(sz, -1)

                p0000 = p0000.reshape(sz, -1)

                p0100 = p0100.reshape(sz, -1)
                p1000 = p1000.reshape(sz, -1)
                p1100 = p1100.reshape(sz, -1)
                fa = fa.reshape(-1, 1)

                p0001 = p0001.reshape(sz, -1)
                p0101 = p0101.reshape(sz, -1)
                p1001 = p1001.reshape(sz, -1)
                p1101 = p1101.reshape(sz, -1)
                fb = fb.reshape(-1, 1)
                fc = fc.reshape(-1, 1)

                p0010 = p0010.reshape(sz, -1)
                p0110 = p0110.reshape(sz, -1)
                p1010 = p1010.reshape(sz, -1)
                p1110 = p1110.reshape(sz, -1)
                fd = fd.reshape(-1, 1)

                p0011 = p0011.reshape(sz, -1)
                p0111 = p0111.reshape(sz, -1)
                p1011 = p1011.reshape(sz, -1)
                p1111 = p1111.reshape(sz, -1)

                fab = fa > fb;
                fac = fa > fc;
                fad = fa > fd

                fbc = fb > fc;
                fbd = fb > fd;
                fcd = fc > fd

                i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
                i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]
                i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]
                i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]

                i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                        p1110[i] + (fd[i]) * p1111[i]
                i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]
                i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
                out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]
                i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]

                i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                        p1110[i] + (fd[i]) * p1111[i]
                # Fix the overflow bug in SR-LUT's implementation, should compare fd with fa first!
                # i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], fab, fcd], dim=1), dim=1)
                # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fa[i]) * p0010[i] + (fa[i]-fd[i]) * p1010[i] + (fd[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
                # i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:,None], ~i10[:,None], fab, fad], dim=1), dim=1)
                # out[i] = (q-fc[i]) * p0000[i] + (fc[i]-fd[i]) * p0010[i] + (fd[i]-fa[i]) * p0011[i] + (fa[i]-fb[i]) * p1011[i] + (fb[i]) * p1111[i]
                i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]
                i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                                    dim=1)  # c > d > a > b
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]
                i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                        p1011[i] + (fb[i]) * p1111[i]

                i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                        p1110[i] + (fd[i]) * p1111[i]
                i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]
                i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]
                i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                        p1101[i] + (fc[i]) * p1111[i]

                i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                        p1110[i] + (fd[i]) * p1111[i]
                i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]
                i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
                out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]
                i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]

                i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                        p1110[i] + (fd[i]) * p1111[i]
                i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]
                i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
                out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]
                i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                                    dim=1)
                out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                        p0111[i] + (fa[i]) * p1111[i]


                out = out.reshape((img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale))
                out = out.permute(0, 1, 2, 4, 3, 5).reshape(
                    (img_a1.shape[0], img_a1.shape[1], img_a1.shape[2] * upscale, img_a1.shape[3] * upscale))

                # 归一化到[-1,1]范围
                # out = out.astype(np.float32) / q / 127.0
                out = out / q
                #out = out.unsqueeze(0)
                out = torch.rot90(out, (4 - r) % 4, [2, 3])
                out_all += out


        final_out = out_all / self.avg_factor
        return final_out