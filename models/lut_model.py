import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import bit_plane_slicing, decode_bit_mask, round_func
from .luts import HDLUT, HDVLUT, HDBLUT, HDBVLUT, HLLUT, HSLUT, \
   PCMComb4inLUT, LLUT, HDBT2LUT, HDBLRC1LUT
import numpy as np


class LUT_Model(nn.Module):
    def __init__(self, msb_weights, lsb_weights, msb='hs', lsb='hs', msb2='hdblrc', lsb2='hdblrc', msb3='hdbl', lsb3='l', upscale=2, pcm_flag=True, pcm_weights=None,
                 lut_weights=None, rot2_weights=None, msb_weights2=None, lsb_weights2=None, msb_weights3=None):
        super(LUT_Model, self).__init__()
        self.upscale = upscale
        self.bit_mask = '11110000'
        self.msb_bits, self.lsb_bits, self.msb_step, self.lsb_step = decode_bit_mask(self.bit_mask)
        unit_dict = {'hd': HDLUT, 'hl': HLLUT, 'hs': HSLUT, 'hdv': HDVLUT, 'hdb': HDBLUT, 'hdbv': HDBVLUT
            ,'hdbl': HDBL2LUT, 'hdblrc': HDBLRC1LUT}
        self.pcm_flag = pcm_flag
        if self.pcm_flag == True:
            self.pcm_lut = PCMComb4inLUT(*pcm_weights)
        # msb
        msb_lut = unit_dict[msb]
        # MSB
        self.msb_lut = msb_lut(*msb_weights, 2 ** self.msb_bits, upscale=upscale)

        # LSB
        lsb_lut = unit_dict[lsb]
        self.lsb_lut = lsb_lut(*lsb_weights, 2 ** self.lsb_bits, upscale=upscale)

        # msb2
        msb_lut2 = unit_dict[msb2]
        # MSB2
        self.msb_lut2 = msb_lut2(*msb_weights2, 2 ** self.msb_bits, upscale=upscale)

        # LSB2
        lsb_lut2 = unit_dict[lsb2]
        self.lsb_lut2 = lsb_lut2(*lsb_weights2, 2 ** self.lsb_bits, upscale=upscale)

        msb_lut3 = unit_dict[msb3]
        self.msb_lut3 = msb_lut3(*msb_weights3, 2 ** self.msb_bits, upscale=upscale)

        self.lsb_lut_rot2 = LLUT(rot2_weights, 2 ** self.lsb_bits, upscale=upscale)

        self.weight_lut_msb = nn.Parameter(
            lut_weights[0]
        )

        self.weight_lut_lsb = nn.Parameter(
            lut_weights[1]
        )

    def forward(self, img_lr):
        if self.pcm_flag == True:
            x_uv = torch.clamp(img_lr[:, 1:, :, :] * 1.5, 0.0, 1.0)
            x_uv = 0.8 * img_lr[:, 1:, :, :] + 0.2 * x_uv
            img_lr = torch.clamp(round_func(torch.cat([img_lr[:, 0:1, :, :], x_uv], dim=1) * 255.0), 0, 255)
            avg_factor, bias, norm = 4, 127, 255.0
            PCM_out = self.pcm_lut(img_lr) / norm
            PCM_out = torch.clamp(PCM_out, -1, 1)
            PCM_out = PCM_out + nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr[:, 0:1, :, :] / norm)
            img_lr[:, 0:1, :, :] = torch.clamp(PCM_out, 0, 1)


        img_lr_255 = torch.clamp(torch.round(img_lr[:, 0:1, :, :] * 255), 0, 255)
        img_lr_msb, img_lr_lsb = bit_plane_slicing(img_lr_255, self.bit_mask)

        # msb
        MSB = img_lr_msb / 255.0
        mean_msb = F.avg_pool2d(MSB, kernel_size=5, stride=1, padding=2)
        variance_msb = F.avg_pool2d((MSB - mean_msb) ** 2, kernel_size=5, stride=1, padding=2)
        var_msb_norm = torch.clamp(variance_msb / 0.01, 0, 1)

        B, C, H, W = MSB.shape
        grid_msb = torch.zeros(B, H, W, 2).to(MSB.device)
        grid_msb[:, :, :, 0] = var_msb_norm.squeeze(1) * 2.0 - 1.0  
        grid_msb[:, :, :, 1] = 0 

        lut_tensor = self.weight_lut_msb.T.view(1, 3, 1, 64)
        weights_msb = F.grid_sample(lut_tensor.expand(B, -1, -1, -1), grid_msb, align_corners=True)

        weights_msb = F.softmax(weights_msb, dim=1)

        w_msb_strong = weights_msb[:, 0:1, :, :]
        w_msb_weak = weights_msb[:, 1:2, :, :]
        w_msb_medium = weights_msb[:, 2:, :, :]

        img_lr_msb = torch.floor_divide(img_lr_msb, self.msb_step)

        MSB_out = self.msb_lut(img_lr_msb) / 255.
        MSB_out = torch.clamp(MSB_out, -1, 1)

        MSB_out_2 = self.msb_lut3(img_lr_msb) / 255.
        MSB_out_2 = torch.clamp(MSB_out_2, -1, 1)

        MSB_out_3 = self.msb_lut2(img_lr_msb) / 255.
        MSB_out_3 = torch.clamp(MSB_out_3, -1, 1)


        MSB_out = w_msb_strong * MSB_out_2 + w_msb_weak * MSB_out + w_msb_medium * MSB_out_3


        # lsb
        LSB = img_lr_lsb/ 255.0
        mean_lsb = F.avg_pool2d(LSB, kernel_size=5, stride=1, padding=2)
        variance_lsb = F.avg_pool2d((LSB - mean_lsb) ** 2, kernel_size=5, stride=1, padding=2)
        var_lsb_norm = torch.clamp(variance_lsb / 0.01, 0, 1)

        B, C, H, W = LSB.shape
        grid_lsb = torch.zeros(B, H, W, 2).to(LSB.device)
        grid_lsb[:, :, :, 0] = var_lsb_norm.squeeze(1) * 2.0 - 1.0
        grid_lsb[:, :, :, 1] = 0 

        lut_tensor = self.weight_lut_lsb.T.view(1, 3, 1, 64)
        weights_lsb = F.grid_sample(lut_tensor.expand(B, -1, -1, -1), grid_lsb, align_corners=True)

        weights_lsb = F.softmax(weights_lsb, dim=1)
        w_lsb_strong = weights_lsb[:, 0:1, :, :]
        w_lsb_weak = weights_lsb[:, 1:2, :, :]
        w_lsb_medium = weights_lsb[:, 2:, :, :]

        img_lr_lsb = torch.floor_divide(img_lr_lsb, self.lsb_step)

        LSB_out = self.lsb_lut(img_lr_lsb) / 255.  
        LSB_out = torch.clamp(LSB_out, -1, 1)

        LSB_out_2 = self.lsb_lut_rot2(img_lr_lsb) / 255.  
        LSB_out_2 = torch.clamp(LSB_out_2, -1, 1)

        LSB_out_3 = self.lsb_lut2(img_lr_lsb) / 255.  
        LSB_out_3 = torch.clamp(LSB_out_3, -1, 1)

        LSB_out = w_lsb_strong * LSB_out_3 + w_lsb_weak * LSB_out_2 + w_lsb_medium * LSB_out


        img_out = MSB_out + LSB_out + nn.Upsample(scale_factor=self.upscale, mode='nearest')(img_lr[:, 0:1, :, :])
        img_out = torch.clamp(img_out, 0, 1)


        return torch.clamp(img_out, 0, 1)

