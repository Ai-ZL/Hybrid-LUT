import os
import argparse
import time
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import cv2

from data_y import SRBenchmark, SIDD_VAL
from utils import PSNR, cal_ssim, cPSNR
from PIL import Image
from utils import seed_everything, _yuv2rgb, _rgb2yuv, _rgb2ycbcr
from models import LUT_Model


device = 'cpu'


def parse_args():
    parser = argparse.ArgumentParser("Testing Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int,  default=8)
    parser.add_argument("--test-dir", type=str, default='/home/dataset/SR/benchmark/',
                        help="Testing images")

    parser.add_argument("--lut-dir", type=str, default='./luts',
                        help="Directory for storing cached LUTs")
    parser.add_argument("--result-dir", type=str, default='./results',
                        help="Directory to store resulted images")
    parser.add_argument("--upscale", nargs='+', type=int, default=[1, 1],
                        help="upscaling factors")
    parser.add_argument('--msb', type=str, default='hs', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb', type=str, default='hs', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--msb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--msb3', type=str, default='hdbl', choices=['p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--lsb3', type=str, default='l', choices=['l','p', 'hl', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl', 'hdblrc', 'hs'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    parser.add_argument('--noise', type=int, default=15, help="default noise level:0, 15, 25, 50")
    parser.add_argument('--window', type=int, default=5, help="Mean filter window")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    if args.noise!=0:
        args.exp_name = "msb_{}_lsb_{}_msb2_{}_lsb2_{}_msb3_{}_lsb3_{}_act_{}_nf_{}_{}".format(args.msb, args.lsb, args.msb2, args.lsb2, args.msb3, args.lsb3, args.act_fn, args.n_filters, factors)
    else:
        args.exp_name = "msb_{}_lsb_{}_msb2_{}_lsb2_{}_msb3_{}_lsb3_{}_act_{}_nf_{}_{}_ft".format(args.msb, args.lsb, args.msb2, args.lsb2, args.msb3, args.lsb3, args.act_fn, args.n_filters, factors)
    args.lut_path = f'{args.lut_dir}/{args.exp_name}'

    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    print(args)


    # Prepare directories
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/{}'.format(args.exp_name)):
        os.mkdir('results/{}'.format(args.exp_name))
        print("Results will be saved to: ", 'results/{}'.format(args.exp_name))



    luts = []
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    models = []

    # Load LUTs
    lut_files = os.listdir(args.lut_path)
    print("LUT path: ", args.lut_path)
    for stage in range(n_stages):
        # msb
        msb_weights = []
        for ktype in args.msb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_MSB_{args.msb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            msb_weights.append(weight)

        # lsb
        lsb_weights = []
        for ktype in args.lsb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_LSB_{args.lsb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            lsb_weights.append(weight)
        
        lut_weights = []
        weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_weight_lut_msb.npy')))
        lut_weights.append(weight)
        weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_weight_lut_lsb.npy')))
        lut_weights.append(weight)

        rot2_weights = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_LSB_2ROT_x1_4bit_int8.npy')).astype(np.int_))

        # msb2
        msb_weights2 = []
        for ktype in args.msb2:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_MSB_{args.msb2.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            msb_weights2.append(weight)

        # lsb2
        lsb_weights2 = []
        for ktype in args.lsb2:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_LSB_{args.lsb2.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            lsb_weights2.append(weight)
        
        # msb3
        msb_weights3 = []
        for ktype in args.msb3:
            weight = torch.tensor(np.load(os.path.join(args.lut_path, f'S{stage}_MSB_{args.msb3.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.int_))
            msb_weights3.append(weight)

        if stage==0:
            pcm_flag=True
            pcm_weights = []

            for ktype in 'abc':
                weight = torch.tensor(np.load(os.path.join(args.lut_path,
                                                           f'S{stage}_PCM_YUV_LUT_{ktype.upper()}_x{args.upscale[stage]}_8bit_int8.npy')).astype(
                    np.int_))
                pcm_weights.append(weight)

            #pcm_weights=None
        else:
            pcm_flag=False
            pcm_weights=None
        models.append(LUT_Model(msb_weights, lsb_weights, msb=args.msb, lsb=args.lsb, msb2=args.msb2, lsb2=args.lsb2, msb=args.msb3, lsb=args.lsb3, upscale=args.upscale[stage], pcm_flag=pcm_flag, 
        pcm_weights=pcm_weights, lut_weights=lut_weights, rot2_weights=rot2_weights,msb_weights2=msb_weights2,lsb_weights2=lsb_weights2,msb_weights3=msb_weights3).to(device))



    # Test datasets
    if args.noise!=0:
        test_loader = SRBenchmark(args.test_dir, scale=sr_scale, noise=args.noise)
        test_datasets = ['CBSD68','Kodak24','Urban100','McMaster']
    else:
        test_loader = SIDD_VAL(args.val_dir, args.noise)
        test_datasets = ['SIDD']

    
    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0

    with torch.no_grad():
        for model in models:
            model.eval()

        for j in range(len(test_datasets)):
            psnrs = []
            ssims = []
            files = test_loader.files[test_datasets[j]]

            best_psnr = 0.0
            best_file=''
            for k in range(len(files)):
                key = test_datasets[j] + '_' + files[k][:-4]

                img_gt = test_loader.ims[key]
                if args.noise!=0:
                    input_im = test_loader.ims[key + 'n%d' % args.noise ]
                else:
                    input_im = test_loader.ims[key + '_noise' ]

                img_gt = _rgb2yuv(img_gt)
                input_im = _rgb2yuv(input_im)


                blur_output = input_im.copy()
                blur_output[:,:,1:] = cv2.blur(blur_output[:,:,1:], (args.window, args.window))

                input_im = input_im.astype(np.float32) / 255.0
                val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(device)  # (1, 3, 128, 128)

                x = val_L

                for model in models:
                    x = model(x)
                

                # Output 
                image_out = (x).cpu().data.numpy()

                image_out = image_out * 255
                image_out = np.transpose(np.clip(image_out[0], 0, 255), [1,2,0]) # BxCxHxW -> HxWxC


                image_out = np.concatenate(
                    [image_out[:, :, 0][:, :, np.newaxis], blur_output[:, :, 1][:, :, np.newaxis],
                     blur_output[:, :, 2][:, :, np.newaxis]], axis=2)


                img_gt = _yuv2rgb(img_gt).astype(np.uint8)
                image_out = _yuv2rgb(image_out).astype(np.uint8)

                y_gt, y_out = img_gt, image_out

                psnrs.append(cPSNR(y_gt, y_out, 0))
                if args.noise==0:
                    ssims.append(cal_ssim(_rgb2ycbcr(y_gt)[:,:,0], _rgb2ycbcr(y_out)[:,:,0]))

                if cPSNR(y_gt, y_out, 0)>best_psnr:
                    best_psnr = cPSNR(y_gt, y_out, 0)
                    best_file = key

            if args.noise!=0:
                print('Dataset {} | AVG LUT PSNR: {:.4f}'.format(test_datasets[j], np.mean(np.asarray(psnrs))))
            else:
                print('Dataset {} | AVG LUT PSNR: {:.4f} SSIM: {:.4f}'.format(test_datasets[j], np.mean(np.asarray(psnrs)), np.mean(np.asarray(ssims))))

            print('best psnr is ', best_psnr, 'the image is ',best_file)


        
        
