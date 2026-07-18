import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import argparse

from models import HKLUT
from data_y import SIDD_VAL, ProviderDN_C

from utils import PSNR, logger_info, _rgb2yuv, seed_everything

torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser("Finetuning Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=8)
    parser.add_argument("--train-dir", type=str, default='/home/dataset/SR/DIV2K',
                        help="Training images")
    parser.add_argument("--val-dir", type=str, default='/home/dataset/SR/benchmark',
                        help="Validation images")

    parser.add_argument("--lut-dir", type=str, default='./luts',
                        help="Directory for storing cached LUTs")
    parser.add_argument("--result-dir", type=str, default='./results',
                        help="Directory to store resulted images")
    parser.add_argument("--upscale", nargs='+', type=int, default=[1, 1],
                        help="upscaling factors")
    parser.add_argument("--crop-size", type=int, default=48,
                        help="input LR training patch size")
    parser.add_argument('--msb', type=str, default='hs',
                        choices=['p', 'hl', 'hs', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb', type=str, default='hs',
                        choices=['p', 'hl', 'hs', 'hd', 'hdb', 'hdv', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--msb2', type=str, default='hdblrc', choices=['p', 'hl', 'hs', 'hd', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hs', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--msb3', type=str, default='hdbl', choices=['p', 'hl', 'hs', 'hd', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb3', type=str, default='l', choices=['l','p', 'hl', 'hd', 'hs', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0,  help='weight decay')
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    parser.add_argument('--noise', type=int, default=15, help="default noise level:15, 25, 50")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Set 0 for from scratch, else will load saved params and trains further")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="training batch size")
    parser.add_argument("--i-display", type=int, default=100,
                        help="display info every N iteration")
    parser.add_argument("--i-validate", type=int, default=500,
                        help="validation every N iteration")
    parser.add_argument('--totalIter', type=int, default=4000, help='Total number of training iterations')
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb_{}_lsb_{}_msb2_{}_lsb2_{}_msb3_{}_lsb3_{}_act_{}_nf_{}_{}_ft".format(args.msb, args.lsb, args.msb2, args.lsb2, args.msb3, args.lsb3, args.act_fn, args.n_filters, factors)

    args.lut_path = f'{args.lut_dir}/{args.exp_name}'

    return args

best_psnr=0
update=0

def save_luts(model_G, args, n_stages):
    for s in range(n_stages):
        stage = s
        
        if stage==0:
            for ktype in 'abc':
                lut_path_pcm = f'luts/{args.exp_name}/LUT_ft_S{stage}_YUV_LUT_{ktype.upper()}_x{args.upscale[stage]}_8bit_int8.npy'
                
                # get pcm_lut
                pcm_lut_obj = getattr(model_G[stage], 'pcm_lut')
                weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight')  # a_weight, b_weight, c_weight
                
                lut_weight = np.round(
                    np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
                ).astype(np.int8)
                
                np.save(lut_path_pcm, lut_weight)
                #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_pcm}")
        
        lut_path_msb = f'luts/{args.exp_name}/LUT_ft_S{stage}_weight_lut_msb.npy'
            
        # get pcm_lut_msb
        pcm_lut_msb = getattr(model_G[stage], "weight_lut_msb")
        lut_weight = pcm_lut_msb.cpu().detach().numpy()
            
        np.save(lut_path_msb, lut_weight)
        #print(f"Stage {stage} weight_lut_msb - LUT size: {lut_weight.shape}, Saved to {lut_path_msb}")

        lut_path_lsb = f'luts/{args.exp_name}/LUT_ft_S{stage}_weight_lut_lsb.npy'
            
        # get pcm_lut_lsb
        pcm_lut_lsb = getattr(model_G[stage], "weight_lut_lsb")
        lut_weight = pcm_lut_lsb.cpu().detach().numpy()
            
        np.save(lut_path_lsb, lut_weight)
        #print(f"Stage {stage} weight_lut_lsb - LUT size: {lut_weight.shape}, Saved to {lut_path_lsb}")

        
        for ktype in args.msb:
            lut_path_msb = f'luts/{args.exp_name}/LUT_ft_S{stage}_MSB_{args.msb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy'
            
            # get msb_lut
            pcm_lut_obj = getattr(model_G[stage], 'msb_lut')
            weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight') 
            
            lut_weight = np.round(
                np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
            ).astype(np.int8)
            
            np.save(lut_path_msb, lut_weight)
            #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_msb}")

            lut_path_lsb = f'luts/{args.exp_name}/LUT_ft_S{stage}_LSB_{args.msb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy'
            
            # get lsb_lut
            pcm_lut_obj = getattr(model_G[stage], 'lsb_lut')
            weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight') 
            lut_weight = np.round(
                np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
            ).astype(np.int8)
            
            np.save(lut_path_lsb, lut_weight)
            #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_lsb}")
        
        for ktype in args.msb2:
            lut_path_msb = f'luts/{args.exp_name}/LUT_ft_S{stage}_MSB_HDBLRC_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy'
            
            # get msb_lut2
            pcm_lut_obj = getattr(model_G[stage], 'msb_lut2')
            weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight') 
            
            lut_weight = np.round(
                np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
            ).astype(np.int8)
            
            np.save(lut_path_msb, lut_weight)
            #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_msb}")

            lut_path_lsb = f'luts/{args.exp_name}/LUT_ft_S{stage}_LSB_HDBLRC_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy'
            
            # get lsb_lut2
            pcm_lut_obj = getattr(model_G[stage], 'lsb_lut2')
            weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight') 
            lut_weight = np.round(
                np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
            ).astype(np.int8)
            
            np.save(lut_path_lsb, lut_weight)
            #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_lsb}")
        
        for ktype in args.msb3:
            lut_path_msb = f'luts/{args.exp_name}/LUT_ft_S{stage}_MSB_HDBT_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy'
            
            # get msb_lut3
            pcm_lut_obj = getattr(model_G[stage], 'msb_lut3')
            weight_attr = getattr(pcm_lut_obj, f'{ktype}_weight') 
            
            lut_weight = np.round(
                np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
            ).astype(np.int8)
            
            np.save(lut_path_msb, lut_weight)
            #print(f"Stage {stage} {ktype}_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_msb}")
        
        lut_path_lsb = f'luts/{args.exp_name}/LUT_ft_S{stage}_LSB_2ROT_x{args.upscale[stage]}_4bit_int8.npy'
            
        
        pcm_lut_obj = getattr(model_G[stage], 'lsb_lut_rot2')
        weight_attr = getattr(pcm_lut_obj, 'l_weight') 
        
        lut_weight = np.round(
            np.clip(weight_attr.cpu().detach().numpy(), -1, 1) * 127
        ).astype(np.int8)
        
        np.save(lut_path_lsb, lut_weight)
        #print(f"Stage {stage} 2ROT l_weight - LUT size: {lut_weight.shape}, Saved to {lut_path_lsb}")

def valid_steps(model_G, valid, args, iter):
    global best_psnr, update
    datasets = ['SIDD']

    with torch.no_grad():
        for model in model_G:
            model.eval()

        for i in range(len(datasets)):
            psnrs = []
            files = valid.files[datasets[i]]

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                img_gt = valid.ims[key]
                input_im = valid.ims[key + '_noise' ]#'n%d' % args.noise

                img_gt = _rgb2yuv(img_gt)
                input_im = _rgb2yuv(input_im)

                input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()
                
                x=im

                for model in model_G:
                    x = model(x)
                
                image_out = (x).cpu().data.numpy()
                image_out = image_out * 255.0  # (1, 3, 512, 512)
                image_out = np.transpose(np.clip(image_out[0], 0, 255), [1, 2, 0])  # BxCxHxW -> HxWxC


                left, right = image_out[:, :, 0], img_gt[:, :, 0]
                psnrs.append(PSNR(left, right, 0))


            logger.info('Iter {} | Dataset {} | AVG PSNR: {:02f}'.format(iter, datasets[i],
                                                                                            np.mean(np.asarray(psnrs))))
            if np.mean(np.asarray(psnrs))>best_psnr:
                best_psnr=np.mean(np.asarray(psnrs))
                update=1


if __name__ == "__main__":

    args = parse_args()
    print(args)
    seed_everything(args.seed)

    ## Prepare directories
    if not os.path.isdir('luts'):
        os.mkdir('luts')
    if not os.path.isdir('luts/{}'.format(args.exp_name)):
        os.mkdir('luts/{}'.format(args.exp_name))


    logger_name = 'lutft'
    logger_info(logger_name, os.path.join(args.lut_path, logger_name + '.log'))
    logger = logging.getLogger(logger_name)


    model_G = []
    i=0
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)

    # Load LUTs
    print("LUT path: ", args.lut_path[:-3])
    for stage in range(n_stages):
        if stage != 0:
            pcm_flag = False
            lut_weights=[]
            lut_weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                        f'S{stage}_weight_lut_msb.npy')))
            lut_weights.append(lut_weight)
            lut_weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                           f'S{stage}_weight_lut_lsb.npy')))
            lut_weights.append(lut_weight)

            rot2_weights = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                           f'S{stage}_LSB_2ROT_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            pcm_weights=None
        else:
            pcm_flag = True
            pcm_weights = []
            for ktype in 'abc':
                weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                           f'S{stage}_PCM_YUV_LUT_{ktype.upper()}_x{args.upscale[stage]}_8bit_int8.npy')).astype(np.float32) / 127.0)
                pcm_weights.append(weight)
            lut_weights = []
            lut_weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                           f'S{stage}_weight_lut_msb.npy')))
            lut_weights.append(lut_weight)
            lut_weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                           f'S{stage}_weight_lut_lsb.npy')))
            lut_weights.append(lut_weight)

            rot2_weights = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                            f'S{stage}_LSB_2ROT_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
        # msb
        msb_weights = []
        for ktype in args.msb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                       f'S{stage}_MSB_{args.msb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            msb_weights.append(weight)

        # lsb
        lsb_weights = []
        for ktype in args.lsb:
            weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                       f'S{stage}_LSB_{args.lsb.upper()}_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            lsb_weights.append(weight)
        
        # msb
        msb_weights2 = []
        for ktype in args.msb2:
            weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                       f'S{stage}_MSB_HDBLRC_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            msb_weights2.append(weight)

        # lsb
        lsb_weights2 = []
        for ktype in args.lsb2:
            weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                       f'S{stage}_LSB_HDBLRC_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            lsb_weights2.append(weight)

        msb_weights3 = []
        for ktype in args.msb3:
            weight = torch.tensor(np.load(os.path.join(args.lut_path[:-3],
                                                       f'S{stage}_MSB_HDBT_LUT_{ktype.upper()}_x{args.upscale[stage]}_4bit_int8.npy')).astype(np.float32) / 127.0)
            msb_weights3.append(weight)

        model_G.append(
            HKLUT(msb_weights, lsb_weights, msb=args.msb, lsb=args.lsb, upscale=args.upscale[stage], pcm_flag=pcm_flag, pcm_weights=pcm_weights,
                      lut_weights=lut_weights, rot2_weights=rot2_weights, msb_weights2=msb_weights2, lsb_weights2=lsb_weights2, msb_weights3=msb_weights3).cuda())


    # Optimizers
    opt_G = optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters()))} for model in model_G], 
                       lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, eps=1e-8, amsgrad=False)
    

    # Learning rate schedule
    lr1=5e-5
    lr_b = lr1 / args.lr
    lr_a = 1 - lr_b
    lf = lambda x: (((1 + math.cos(x * math.pi / args.totalIter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    # Training dataset
    train_iter = ProviderDN_C(args.batch_size, args.n_workers, sr_scale, args.train_dir, args.crop_size, 0)

    # Valid dataset
    valid = SIDD_VAL(args.val_dir, args.noise)


    # Training
    l_accum = [0., 0., 0.]
    dT = 0.
    rT = 0.
    accum_samples = 0
    i = args.start_iter
    for i in range(args.start_iter + 1, args.totalIter + 1):
        for model in model_G:
            model.train()
        

        # Data preparing
        st = time.time()
        im, lb = train_iter.next()
        im = im.cuda()
        lb = lb.cuda()
        dT += time.time() - st

        st = time.time()
        opt_G.zero_grad()

        x = im
        for model in model_G:
            x = model(x)
        pred = torch.clamp(x, 0, 1)

        loss_G = F.mse_loss(pred[:, 0:1, :, :], lb[:, 0:1, :, :])
        loss_G.backward()
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += args.batch_size
        l_accum[0] += loss_G.item()

        # Show information
        if i % args.i_display == 0:
            logger.info("{} | Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                args.exp_name, i, accum_samples, l_accum[0] / args.i_display, dT / args.i_display,
                                              rT / args.i_display))
            l_accum = [0., 0., 0.]
            dT = 0.
            rT = 0.

        # Validation
        if (i % args.i_validate == 0) or (i == 1):
            # Validation during multi GPU training
            valid_steps(model_G, valid, args, i)
            print(best_psnr," psnr---update ",update)
        
        if update==1:
            save_luts(model_G, args, n_stages)
            print("Save luts")
            update=0

    # Save finetuned LUTs
    logger.info("Finetuned LUT saved to {}".format(args.lut_dir))
    logger.info("Complete")
