import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import os
from tqdm import tqdm
import argparse
import math

from torch.utils.tensorboard import SummaryWriter

from models import *
from data_y import Provider, SRBenchmark, ProviderDN_C, SIDD_VAL
from utils import PSNR, seed_everything, _rgb2yuv


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser("Training Setting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int,  default=8)
    parser.add_argument("--train-dir", type=str, default='/dataset/SR/DIV2K',
                        help="Training images")
    parser.add_argument("--val-dir", type=str, default='/dataset/SR/benchmark',
                        help="Validation images")
    parser.add_argument("--i-display", type=int, default=500,
                        help="display info every N iteration")
    parser.add_argument("--i-validate", type=int, default=500,
                        help="validation every N iteration")
    parser.add_argument("--i-save", type=int, default=5000,
                        help="save checkpoints every N iteration")

    parser.add_argument("--upscale", nargs='+', type=int, default=[2, 2],
                        help="upscaling factors")
    parser.add_argument("--crop-size", type=int, default=48,
                        help="input LR training patch size")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="training batch size")
    parser.add_argument("--start-iter", type=int, default=0,
                        help="Set 0 for from scratch, else will load saved params and train further")
    parser.add_argument("--train-iter", type=int, default=200000,
                        help="number of training iterations")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--wd', type=float, default=0,  help='weight decay')

    parser.add_argument('--msb', type=str, default='hs', choices=['p', 'hl', 'hs', 'hd', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb', type=str, default='hs', choices=['p', 'hl', 'hd', 'hs', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--msb2', type=str, default='hdblrc', choices=['p', 'hl', 'hs', 'hd', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb2', type=str, default='hdblrc', choices=['p', 'hl', 'hd', 'hs', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--msb3', type=str, default='hdbl', choices=['p', 'hl', 'hs', 'hd', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--lsb3', type=str, default='l', choices=['l','p', 'hl', 'hd', 'hs', 'sdy', 'hdv', 'hdb', 'hdbv', 'hdbl','hdblrc'])
    parser.add_argument('--act-fn', type=str, default='relu', choices=['relu', 'gelu', 'leakyrelu', 'starrelu'])
    parser.add_argument('--n-filters', type=int, default=64, help="number of filters in intermediate layers")
    parser.add_argument('--noise', type=int, default=15, help="default noise level: 0, 15, 25, 50")
    args = parser.parse_args()

    factors = 'x'.join([str(s) for s in args.upscale])
    args.exp_name = "msb_{}_lsb_{}_msb2_{}_lsb2_{}_msb3_{}_lsb3_{}_act_{}_nf_{}_{}".format(args.msb, args.lsb, args.msb2, args.lsb2, args.msb3, args.lsb3, args.act_fn, args.n_filters, factors)

    act_fn_dict = {'relu': nn.ReLU, 'gelu': nn.GELU, 'leakyrelu': nn.LeakyReLU, 'starrelu': StarReLU}
    args.act_fn = act_fn_dict[args.act_fn]

    return args


def SaveCheckpoint(models, opt_G, i, args, best=False):
    if best:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 'checkpoint/{}/model_G_S{}_best.pth'.format(args.exp_name, stage))
            else:
                torch.save(model.state_dict(), 'checkpoint/{}/model_G_S{}_best.pth'.format(args.exp_name, stage))
        torch.save(opt_G.state_dict(), 'checkpoint/{}/opt_G_best.pth'.format(args.exp_name))
        torch.save(scheduler.state_dict(), 'checkpoint/{}/sch_G_best.pth'.format(args.exp_name))
        print("Best checkpoint saved")
    else:
        for stage, model in enumerate(models):
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), 'checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, i))
            else:
                torch.save(model.state_dict(), 'checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, i))
        torch.save(opt_G.state_dict(), 'checkpoint/{}/opt_G_i{:06d}.pth'.format(args.exp_name, i))
        torch.save(scheduler.state_dict(), 'checkpoint/{}/sch_G_i{:06d}.pth'.format(args.exp_name, i))
        print("Checkpoint saved {}".format(str(i)))


if __name__ == "__main__":
    args = parse_args()
    print(args)
    seed_everything(args.seed)

    ### Tensorboard for monitoring ###
    writer = SummaryWriter(log_dir='./log/{}'.format(args.exp_name))

    models = []
    i=0
    n_stages = len(args.upscale)
    sr_scale = np.prod(args.upscale)
    pcm_flag=True

    for s in args.upscale:
        if i!=0:
            pcm_flag=False
        models.append(NET_MODEL(msb=args.msb, lsb=args.lsb, msb2=args.msb2, lsb2=args.lsb2, msb3=args.msb3, lsb3=args.lsb3, nf=args.n_filters, upscale=s, act=args.act_fn, pcm_flag=pcm_flag).to(device))
        i+=1

    if torch.cuda.device_count() > 1:
        models = [nn.DataParallel(model) for model in models]

    ## Optimizers
    opt_G = optim.Adam([{'params': list(filter(lambda p: p.requires_grad, model.parameters()))} for model in models], 
                       lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd, eps=1e-8, amsgrad=False)

    lr1=5e-5
    lr_b = lr1 / args.lr
    lr_a = 1 - lr_b
    lf = lambda x: (((1 + math.cos(x * math.pi / args.train_iter)) / 2) ** 1.0) * lr_a + lr_b
    scheduler = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lf)

    ## Load saved params
    if args.start_iter > 0:
        for stage in range(n_stages):
            lm = torch.load('checkpoint/{}/model_G_S{}_i{:06d}.pth'.format(args.exp_name, stage, args.start_iter))
            models[stage].load_state_dict(lm, strict=True)

        lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(args.exp_name, args.start_iter))
        opt_G.load_state_dict(lm)
    
    
    # Training dataset
    if args.noise==0:
        train_loader = ProviderDN_C(args.batch_size, args.n_workers, sr_scale, args.train_dir, args.crop_size, args.noise)#SIDD
    else:
        train_loader = Provider(args.batch_size, args.n_workers, sr_scale, args.train_dir, args.crop_size, noise=args.noise)

    # Validation dataset
    if args.noise==0:
        valid_loader = SIDD_VAL(args.val_dir, args.noise)
    else:
        valid_loader = SRBenchmark(args.val_dir, scale=sr_scale, noise = args.noise)
    
    if args.noise==0:
        valid_datasets = ['SIDD']
    else:
        valid_datasets = ['CBSD68']

    ## Prepare directories
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir('checkpoint/{}'.format(args.exp_name)):
        os.mkdir('checkpoint/{}'.format(args.exp_name))
    if not os.path.isdir('log'):
        os.mkdir('log')

    l_accum = [0.,0.,0.]
    dT = 0.
    rT = 0.
    accum_samples = 0


    ### TRAINING
    best_psnr = 0
    best_psnr_dataset = [0 for _ in range(len(valid_datasets))]
    for i in tqdm(range(args.start_iter+1, args.train_iter+1)):

        for model in models:
            model.train()

        # Data preparing
        st = time.time()
        batch_L, batch_H = train_loader.next()
        batch_H = batch_H.to(device)      # BxCxHxW (32, 3, 192, 192), range [0,1]
        batch_L = batch_L.to(device)      # BxCxHxW (32, 3, 48, 48), range [0,1]
    
        dT += time.time() - st


        ## TRAIN G
        st = time.time()
        opt_G.zero_grad()

        x = batch_L
        for model in models:
            x = model(x)

        pred = torch.clamp(x, 0, 1)  # [-2, 2] -> [0, 1]
        loss_G = F.mse_loss(pred[:, 0:1, :, :], batch_H[:, 0:1, :, :])

        # Update
        loss_G.backward()
        # --- 梯度裁剪 ---
        for model in models:
            module = model.module if isinstance(model, nn.DataParallel) else model
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
        opt_G.step()
        scheduler.step()

        rT += time.time() - st

        # For monitoring
        accum_samples += args.batch_size
        l_accum[0] += loss_G.item()


        ## Show information
        if i % args.i_display == 0:
            writer.add_scalar('loss_Pixel', l_accum[0]/args.i_display, i)
            print("{}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
                args.exp_name, i, accum_samples, l_accum[0]/args.i_display, dT/args.i_display, rT/args.i_display))
            l_accum = [0.,0.,0.]
            dT = 0.
            rT = 0.


        ## Save models
        if i % args.i_save == 0:
            SaveCheckpoint(models, opt_G, i, args)


        ## Validation
        if i % args.i_validate == 0:
            with torch.no_grad():
                for model in models:
                    model.eval()


                for j in range(len(valid_datasets)):
                    psnrs = []
                    files = valid_loader.files[valid_datasets[j]]

                    for k in range(len(files)):
                        key = valid_datasets[j] + '_' + files[k][:-4]

                        img_gt = valid_loader.ims[key] # (512, 512, 3) range [0, 255]
                        if args.noise==0:
                            input_im = valid_loader.ims[key + '_noise'] # (128, 128, 3) range [0, 255]
                        else:
                            input_im = valid_loader.ims[key + 'n%d' % args.noise] # (128, 128, 3) range [0, 255]   

                        img_gt = _rgb2yuv(img_gt)
                        input_im = _rgb2yuv(input_im)


                        input_im = input_im.astype(np.float32) / 255.0  
                        val_L = torch.Tensor(np.expand_dims(np.transpose(input_im, [2, 0, 1]), axis=0)).to(device) # (1, 3, 128, 128)


                        x = val_L
                        for model in models:
                            x = model(x)
                            

                        # Output 
                        image_out = (x).cpu().data.numpy() # (1, 3, 512, 512)
                        image_out = np.transpose(np.clip(image_out[0], 0. , 1.), [1,2,0]) # BxCxHxW -> HxWxC
                        image_out = ((image_out)*255)


                        # PSNR on Y channel
                        psnrs.append(PSNR(img_gt[:,:,0],image_out[:,:,0],0))

                    mean_psnr = np.mean(np.asarray(psnrs))

                    # save best psnr for dataset
                    if mean_psnr > best_psnr:
                        best_psnr = np.mean(np.asarray(psnrs))
                        SaveCheckpoint(models, opt_G, i, args, best=True)
                    if mean_psnr > best_psnr_dataset[j]:
                        best_psnr_dataset[j] = np.mean(np.asarray(psnrs))

                    print('Iter {} | Dataset {} | AVG Val PSNR: {:02f}'.format(i, valid_datasets[j], mean_psnr))
                    writer.add_scalar('PSNR_valid/{}'.format(valid_datasets[j]), mean_psnr, i)
                    writer.flush()

    print(f'Best PSNR: {best_psnr}')
    for j in range(len(valid_datasets)):
        print('Best PSNR for ',valid_datasets[j]," : ",best_psnr_dataset[j])
