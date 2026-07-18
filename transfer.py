import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from models import *
from utils import decode_bit_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser("Transfer Setting")
    parser.add_argument("--ckpt-dir", type=str, default='./checkpoint',
                        help="Checkpoint directory")
    parser.add_argument("--upscale", nargs='+', type=int, default=[2, 2],
                        help="upscaling factors")
    parser.add_argument('--msb', type=str, default='hdb',
                        choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb', type=str, default='hdv',
                        choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--msb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--msb3', type=str, default='hdbl', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb3', type=str, default='l', choices=['l','p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb_{}_lsb_{}_msb2_{}_lsb2_{}_msb3_{}_lsb3_{}_act_{}_nf_{}_{}".format(args.msb, args.lsb, args.msb2, args.lsb2, args.msb3, args.lsb3, args.act_fn, args.n_filters, factors)

    act_fn_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]
    return args

def get_input_tensor_4d(interval=4):
    # 1D input
    base = torch.arange(0, 257, 2 ** interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 1, 4).float() / 255.0
    return input_tensor


def get_input_tensor(bits, base_steps, n_pixels=3):
    L = 2 ** bits
    base_step_ind = torch.arange(0, L, 1)
    base = base_steps * base_step_ind / 255.0
    index_nD = torch.meshgrid(*[base for _ in range(n_pixels)])
    input_tensor = torch.cat([index_nD[i].flatten().unsqueeze(1) for i in range(len(index_nD))], 1).unsqueeze(1)
    return input_tensor  # N, 1, n_pixels

def get_input_tensor_1d(interval=4):
    q = 2 ** interval
    base = torch.arange(0, 255 + q, q)#torch.arange(0, 256 + q, q)
    base[-1] = 255
    x = base.float() / 255.0
    return x.view(-1, 1, 1, 1)   # [L,1,1,1]



pixel_dict = {'hdb': 3, 'hdv': 2, 'hdbv': 3, 'hdbl': 3, 'hdblrc': 3, 'hd': 2, 'hl': 3, 'hs': 3, 'p': 1}

if __name__ == "__main__":
    args = parse_args()
    print(args)

    models = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    for i, s in enumerate(args.upscale):
        if i==0:
            pcm_flag=True
        else:
            pcm_flag=False
        model = HKNet(msb=args.msb, lsb=args.lsb, msb2=args.msb2, lsb2=args.lsb2, msb3=args.msb3, lsb3=args.lsb3, nf=args.n_filters, upscale=s, act=args.act_fn, pcm_flag=pcm_flag).to(device)
        ckpt = torch.load(f'{args.ckpt_dir}/{args.exp_name}/model_G_S{i}_best.pth')
        model.load_state_dict(ckpt, strict=True)
        models.append(model)

    ## Prepare directories
    if not os.path.isdir('luts'):
        os.mkdir('luts')
    if not os.path.isdir('luts/{}'.format(args.exp_name)):
        os.mkdir('luts/{}'.format(args.exp_name))

    # Extract input-output pairs
    pcm_bits = 8
    pcm_interval = 4
    msb_bits, lsb_bits, msb_step, lsb_step = decode_bit_mask('11110000')

    with torch.no_grad():
        for stage in range(n_stages):
            model = models[stage]
            model.eval()
            if hasattr(model, 'pcm_yuv_lut_a') and model.pcm_flag:
                pcm_luts = [f'pcm_yuv_lut_{ktype}' for ktype in
                            'abc']  # filter(lambda a: 'pcm' in a and 'lut' in a, dir(model))

                for lut in pcm_luts:
                    pcm_unit = model.__getattr__(lut)

                    lut_input = get_input_tensor_4d(pcm_interval)

                    lut_input = pcm_unit.get_lut_input(lut_input).to(device)

                    # Split input to not over GPU memory
                    B = lut_input.size(0) // 100
                    outputs = []

                    for b in range(100):
                        if b == 99:
                            batch_input = lut_input[b * B:]
                        else:
                            batch_input = lut_input[b * B:(b + 1) * B]

                        batch_output = pcm_unit(batch_input)

                        results = torch.round(torch.clamp(batch_output, -1, 1) * 127).cpu().data.numpy().astype(np.int8)
                        outputs += [results]

                    results = np.concatenate(outputs, 0)

                    lut_path_pcm = f'luts/{args.exp_name}/S{stage}_{lut.upper()}_x{args.upscale[stage]}_{pcm_bits}bit_int8.npy'
                    np.save(lut_path_pcm, results)
                    print("Resulting LUT size: ", results.shape, "Saved to", lut_path_pcm)

            gate_luts = filter(lambda a: 'gate' in a, dir(model))
            for lut in gate_luts:
                gate_unit = model.__getattr__(lut)
                lut_input = get_input_tensor_1d(0).to(device)
                y = gate_unit(lut_input)
                lut_out = torch.round(torch.clamp(y, 0, 1) * 255).cpu().data.numpy().astype(np.uint8)
                #print(lut_out[255])
                lut_path_gate = f'luts/{args.exp_name}/S{stage}_{lut.upper()}_x{args.upscale[stage]}_{pcm_bits}bit_int8.npy'
                np.save(lut_path_gate, lut_out)
                print("Resulting LUT size: ", lut_out.shape, "Saved to", lut_path_gate)



            msb_luts = filter(lambda a: 'msb' in a and (args.msb in a or '2rot' in a or 'hdbl' in a or 'hdblrc' in a), dir(model))
            lsb_luts = filter(lambda a: 'lsb' in a and (args.lsb in a or '2rot' in a or 'hdbl' in a or 'hdblrc' in a), dir(model))

            # msb
            for lut in msb_luts:
                msb_unit = model.__getattr__(lut)
                lut_input = get_input_tensor(msb_bits, msb_step, n_pixels=pixel_dict[args.msb])

                lut_input = msb_unit.get_lut_input(lut_input).to(device)

                # Split input to not over GPU memory
                B = lut_input.size(0) // 100
                outputs = []

                for b in range(100):
                    if b == 99:
                        batch_input = lut_input[b * B:]
                    else:
                        batch_input = lut_input[b * B:(b + 1) * B]

                    batch_output = msb_unit(batch_input)

                    results = torch.floor(torch.clamp(batch_output, -1, 1) * 127).cpu().data.numpy().astype(np.int8)
                    outputs += [results]

                results = np.concatenate(outputs, 0)

                lut_path_msb = f'luts/{args.exp_name}/S{stage}_{lut.upper()}_x{args.upscale[stage]}_{msb_bits}bit_int8.npy'
                np.save(lut_path_msb, results)
                print("Resulting LUT size: ", results.shape, "Saved to", lut_path_msb)

            # lsb
            for lut in lsb_luts:
                lsb_unit = model.__getattr__(lut)
                lut_input = get_input_tensor(lsb_bits, lsb_step, n_pixels=pixel_dict[args.lsb])

                lut_input = lsb_unit.get_lut_input(lut_input).to(device)

                # Split input to not over GPU memory
                B = lut_input.size(0) // 100
                outputs = []

                for b in range(100):
                    if b == 99:
                        batch_input = lut_input[b * B:]
                    else:
                        batch_input = lut_input[b * B:(b + 1) * B]

                    batch_output = lsb_unit(batch_input)

                    results = torch.floor(torch.clamp(batch_output, -1, 1) * 127).cpu().data.numpy().astype(np.int8)
                    outputs += [results]

                results = np.concatenate(outputs, 0)

                lut_path_lsb = f'luts/{args.exp_name}/S{stage}_{lut.upper()}_x{args.upscale[stage]}_{lsb_bits}bit_int8.npy'
                np.save(lut_path_lsb, results)
                print("Resulting LUT size: ", results.shape, "Saved to", lut_path_lsb)


            arr_msb = model.__getattr__('weight_lut_msb').cpu().numpy()  # (64, 2)
            lut_path_msb = f'luts/{args.exp_name}/S{stage}_weight_lut_msb.npy'
            np.save(lut_path_msb, arr_msb)
            print("Resulting parameter LUT size: ", arr_msb.shape, "Saved to", lut_path_msb)

            arr_lsb = model.__getattr__('weight_lut_lsb').cpu().numpy()  # (64, 2)
            lut_path_lsb = f'luts/{args.exp_name}/S{stage}_weight_lut_lsb.npy'
            np.save(lut_path_lsb, arr_lsb)
            print("Resulting parameter LUT size: ", arr_lsb.shape, "Saved to", lut_path_lsb)



