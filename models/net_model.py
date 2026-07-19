import torch
import torch.nn as nn
from .units import *
from utils import bit_plane_slicing, floor_func, round_func, _yuv2rgb
import cv2


class NET_MODEL(nn.Module):
    def __init__(self, msb='hs', lsb='hs', msb2='hdblrc', lsb2='hdblrc', msb3='hdbl', lsb3='l', nf=64, upscale=2, act=nn.GELU, pcm_flag=True, **kwargs):
        super(NET_MODEL, self).__init__()
        self.msb = msb
        self.lsb = lsb
        self.msb2 = msb2
        self.lsb2 = lsb2
        self.msb3 = msb3
        self.lsb3 = lsb3
        self.nf = nf
        self.upscale = upscale
        self.act = act
        self.pcm_flag=pcm_flag
        if self.pcm_flag==True:
            self.pcm = PCM_comb_Module_4in
            self.pcm_rot_dict = PCM_comb_Module_4in.rot_dict
            self.pcm_pad_dict = PCM_comb_Module_4in.pad_dict
            self.pcm_avg_factor = PCM_comb_Module_4in.avg_factor
            for ktype in 'abc':
                setattr(self, f'pcm_yuv_lut_{ktype}', self.pcm(ktype=ktype, nf=nf))
        unit_dict = {'p': PUnit, 'hd': HDUnit, 'hs': HSUnit, 'hl': HLUnit, 'hdv': HDVUnit, 'hdb': HDBUnit, 'sdy': SDYUnit, 'hdbv': HDBVUnit, 
        'hdbl': HDBL2Unit, 'hdblrc':HDBLRC1Unit}

        # msb
        msb_unit = unit_dict[msb]
        self.msb_rot_dict = msb_unit.rot_dict
        self.msb_pad_dict = msb_unit.pad_dict
        self.msb_avg_factor = msb_unit.avg_factor
        for ktype in msb:
            setattr(self, f'msb_{msb}_lut_{ktype}', msb_unit(ktype=ktype, nf=nf, upscale=upscale, act=act))

        # lsb
        lsb_unit = unit_dict[lsb]
        self.lsb_rot_dict = lsb_unit.rot_dict
        self.lsb_pad_dict = lsb_unit.pad_dict
        self.lsb_avg_factor = lsb_unit.avg_factor
        for ktype in lsb:
            setattr(self, f'lsb_{lsb}_lut_{ktype}', lsb_unit(ktype=ktype, nf=nf, upscale=upscale, act=act))
        
        # msb
        msb_unit_2 = unit_dict[msb2]
        self.msb_rot_dict2 = msb_unit_2.rot_dict
        self.msb_pad_dict2 = msb_unit_2.pad_dict
        self.msb_avg_factor2 = msb_unit_2.avg_factor
        for ktype in self.msb2:
            setattr(self, f'msb_{self.msb2}_lut_{ktype}', msb_unit_2(ktype=ktype, nf=nf, upscale=upscale, act=act))

        # lsb
        lsb_unit_2 = unit_dict[lsb2]
        self.lsb_rot_dict2 = lsb_unit_2.rot_dict
        self.lsb_pad_dict2 = lsb_unit_2.pad_dict
        self.lsb_avg_factor2 = lsb_unit_2.avg_factor
        for ktype in self.lsb2:
            setattr(self, f'lsb_{self.lsb2}_lut_{ktype}', lsb_unit_2(ktype=ktype, nf=nf, upscale=upscale, act=act))

        msb_unit_3 = unit_dict[msb3]
        self.msb_rot_dict3 = msb_unit_3.rot_dict
        self.msb_pad_dict3 = msb_unit_3.pad_dict
        self.msb_avg_factor3 = msb_unit_3.avg_factor
        for ktype in self.msb3:
            setattr(self, f'msb_{self.msb3}_lut_{ktype}', msb_unit_3(ktype=ktype, nf=nf, upscale=upscale, act=act))

        self.lsb_2rot = unit_dict[lsb3](nf=nf, upscale=upscale, act=act)

        self.weight_lut_msb = nn.Parameter(torch.rand(64, 3)) 
        self.weight_lut_lsb = nn.Parameter(torch.rand(64, 3)) 
        
    def lut_forward(self, x, branch, ktype):
        unit = self.msb if branch == 'msb' else self.lsb
        lut = getattr(self, f'{branch}_{unit}_lut_{ktype}')
        return lut(x)
    
    def lut_forward2(self, x, branch, ktype):
        unit = self.msb2 if branch == 'msb' else self.lsb2
        lut = getattr(self, f'{branch}_{unit}_lut_{ktype}')
        return lut(x)
    
    def lut_forward3(self, x, branch, ktype):
        unit = self.msb3
        lut = getattr(self, f'{branch}_{unit}_lut_{ktype}')
        return lut(x)
    
    def pcm_lut_forward(self, x, ktype, branch):
        lut = getattr(self, f'pcm_{branch}_lut_{ktype}')
        return lut(x)
    

        
    def forward(self, x):
        if self.pcm_flag==True:
            avg_factor, bias, norm = 4, 127, 255.0
            x_uv = torch.clamp(x[:, 1:, :, :]*1.5, 0.0, 1.0)
            x_uv = 0.8 * x[:, 1:, :, :] + 0.2 * x_uv
            x = torch.clamp(round_func(torch.cat([x[:, 0:1, :, :], x_uv], dim=1) * 255), 0, 255)
            x = x / 255.0
            
            pcm_out = 0.0
            for ktype in 'abc':
                for r in self.pcm_rot_dict[ktype]:
                    batch = self.pcm_lut_forward(F.pad(torch.rot90(x, r, [2, 3]), self.pcm_pad_dict[ktype], mode='replicate'), ktype=ktype, branch='yuv')
                    batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                    pcm_out += round_func(batch)
            pcm_out = pcm_out / self.pcm_avg_factor / 255.
            pcm_out = torch.clamp(pcm_out, -1, 1)


            pcm_out += nn.Upsample(scale_factor=self.upscale, mode='nearest')(x[:, 0:1, :, :])
            x = torch.clamp(pcm_out, 0, 1)


        batch_L255 = torch.clamp(round_func(x * 255), 0, 255)
        # Prepare inputs for two branches
        MSB, LSB = bit_plane_slicing(batch_L255, bit_mask='11110000')

        MSB = MSB / 255.0
        LSB = LSB / 255.0

        bias = 127.0
        avg_rot2 = 2

        # MSB
        MSB_out = 0.0
        for ktype in self.msb:
            for r in self.msb_rot_dict[ktype]:
                batch = self.lut_forward(F.pad(torch.rot90(MSB, r, [2, 3]), self.msb_pad_dict[ktype], mode='replicate'), branch='msb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                MSB_out += floor_func(batch)

        MSB_out = MSB_out / self.msb_avg_factor / 255.
        MSB_out = torch.clamp(MSB_out, -1, 1)

        MSB_out_2 = 0.0
        for ktype in self.msb2:
            for r in self.msb_rot_dict2[ktype]:
                batch = self.lut_forward2(F.pad(torch.rot90(MSB, r, [2, 3]), self.msb_pad_dict2[ktype], mode='replicate'), branch='msb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                MSB_out_2 += floor_func(batch)

        MSB_out_2 = MSB_out_2 / self.msb_avg_factor2 / 255.
        MSB_out_2 = torch.clamp(MSB_out_2, -1, 1)

        MSB_out_3 = 0.0
        for ktype in self.msb3:
            for r in self.msb_rot_dict3[ktype]:
                batch = self.lut_forward3(F.pad(torch.rot90(MSB, r, [2, 3]), self.msb_pad_dict3[ktype], mode='replicate'), branch='msb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                MSB_out_3 += floor_func(batch)

        MSB_out_3 = MSB_out_3 / self.msb_avg_factor3 / 255.
        MSB_out_3 = torch.clamp(MSB_out_3, -1, 1)


        mean_msb = F.avg_pool2d(MSB, kernel_size=5, stride=1, padding=2)
        variance_msb = F.avg_pool2d((MSB - mean_msb) ** 2, kernel_size=5, stride=1, padding=2)
        var_msb_norm = torch.clamp(variance_msb / 0.01, 0, 1)

        B, C, H, W = MSB.shape
        grid_msb = torch.zeros(B, H, W, 2).to(MSB.device)
        grid_msb[:, :, :, 0] = var_msb_norm.squeeze(1) * 2.0 - 1.0  # x variance
        grid_msb[:, :, :, 1] = 0  # y fix for 1D LUT

        lut_tensor = self.weight_lut_msb.T.view(1, 3, 1, 64) 
        weights_msb = F.grid_sample(lut_tensor.expand(B, -1, -1, -1), grid_msb, align_corners=True)

        weights_msb = F.softmax(weights_msb, dim=1)
        w_msb_strong = weights_msb[:, 0:1, :, :]
        w_msb_weak = weights_msb[:, 1:2, :, :]
        w_msb_medium = weights_msb[:, 2:, :, :]


        MSB_out = w_msb_weak * MSB_out + w_msb_medium * MSB_out_2 + w_msb_strong * MSB_out_3

        # LSB
        LSB_out = 0.0
        for r in [0, 2]:
            batch = self.lsb_2rot(F.pad(torch.rot90(LSB, r, [2, 3]), (0, 2, 0, 2), mode='replicate'))
            batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
            LSB_out += floor_func(batch)

        LSB_out = LSB_out / avg_rot2 / 255.
        LSB_out = torch.clamp(LSB_out, -1, 1)

        LSB_out_2 = 0.0
        for ktype in self.lsb:
            for r in self.lsb_rot_dict[ktype]:
                batch = self.lut_forward(F.pad(torch.rot90(LSB, r, [2,3]), self.lsb_pad_dict[ktype], mode='replicate'), branch='lsb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                LSB_out_2 += floor_func(batch)
        LSB_out_2 = LSB_out_2 / self.lsb_avg_factor / 255.
        LSB_out_2 = torch.clamp(LSB_out_2, -1, 1)


        LSB_out_3 = 0.0
        for ktype in self.lsb2:
            for r in self.lsb_rot_dict2[ktype]:
                batch = self.lut_forward2(F.pad(torch.rot90(LSB, r, [2,3]), self.lsb_pad_dict2[ktype], mode='replicate'), branch='lsb', ktype=ktype)
                batch = torch.rot90(batch, (4 - r) % 4, [2, 3]) * bias
                LSB_out_3 += floor_func(batch)
        LSB_out_3 = LSB_out_3 / self.lsb_avg_factor2 / 255.
        LSB_out_3 = torch.clamp(LSB_out_3, -1, 1)

        mean_lsb = F.avg_pool2d(LSB, kernel_size=5, stride=1, padding=2)
        variance_lsb = F.avg_pool2d((LSB - mean_lsb) ** 2, kernel_size=5, stride=1, padding=2)
        var_lsb_norm = torch.clamp(variance_lsb / 0.01, 0, 1)

        B, C, H, W = LSB.shape
        grid_lsb = torch.zeros(B, H, W, 2).to(LSB.device)
        grid_lsb[:, :, :, 0] = var_lsb_norm.squeeze(1) * 2.0 - 1.0  # x variance
        grid_lsb[:, :, :, 1] = 0  #  y fix for 1D LUT

        lut_tensor = self.weight_lut_lsb.T.view(1, 3, 1, 64) 
        weights_lsb = F.grid_sample(lut_tensor.expand(B, -1, -1, -1), grid_lsb, align_corners=True)

        weights_lsb = F.softmax(weights_lsb, dim=1)
        w_lsb_strong = weights_lsb[:, 0:1, :, :]
        w_lsb_weak = weights_lsb[:, 1:2, :, :]
        w_lsb_medium = weights_lsb[:, 2:, :, :]

        LSB_out = w_lsb_weak * LSB_out + w_lsb_medium * LSB_out_2 + w_lsb_strong * LSB_out_3

        output = MSB_out + LSB_out
        output += nn.Upsample(scale_factor=self.upscale, mode='nearest')(x)

        return torch.clamp(output, 0, 1)
