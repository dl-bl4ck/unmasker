import json
import os
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adadelta, Adam
from torch.nn import BCELoss, DataParallel
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm.auto import tqdm
from models import CompletionNetwork, ContextDiscriminator
import wandb
import pickle
from utils import (
    gen_input_mask,
    gen_hole_area,
    crop,
    sample_random_batch,
)
from dataset import masked_dataset



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default="./dataset/")
parser.add_argument('--wandb',type=str,default="tmp")
parser.add_argument('--max_train',type=int,default=1000)
parser.add_argument('--max_test',type=int,default=100)
parser.add_argument('--result_dir',type=str, default="./results/")
parser.add_argument('--init_model_cn', type=str, default=None)
parser.add_argument('--init_model_cd', type=str, default=None)
parser.add_argument('--steps_1', type=int, default=1000)
parser.add_argument('--steps_2', type=int, default=1000)
parser.add_argument('--steps_3', type=int, default=1000)
parser.add_argument('--snaperiod_1', type=int, default=10)
parser.add_argument('--snaperiod_2', type=int, default=100)
parser.add_argument('--snaperiod_3', type=int, default=100)
parser.add_argument('--num_test_completions', type=int, default=64)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--batch_size',type=int,default=64)


        
def main(args):
   
    if args.wandb != "tmp":
        wandb.init(project=args.wandb, config=args)

    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available.')
    gpu = torch.device('cuda:0')

    # create result directory (if necessary)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(args.result_dir, phase)):
            os.makedirs(os.path.join(args.result_dir, phase))

    # load dataset
    
    print('loading dataset... (it may take a few minutes)')
    train_dset = masked_dataset(os.path.join(args.data_dir, 'train'), args.max_train)

    test_dset = masked_dataset(os.path.join(args.data_dir, 'test'), args.max_test)
    
    train_loader = DataLoader(
        train_dset,
        batch_size=(args.batch_size),
        shuffle=True)

    alpha = torch.tensor(
        args.alpha,
        dtype=torch.float32).to(gpu)


    model_cn = CompletionNetwork()
    if args.init_model_cn is not None:
        model_cn.load_state_dict(torch.load(
            args.init_model_cn,
            map_location='cpu'))
    
    model_cn = model_cn.to(gpu)
    model_cn.train()
    opt_cn = Adadelta(model_cn.parameters())

    # training
    # ================================================
    # Training Phase 1
    # ================================================
    pbar = tqdm(total=args.steps_1)
    

    while pbar.n < args.steps_1:
        for i, (normal, masked) in enumerate(train_loader, 0):
            # forward
            # normal = torch.autograd.Variable(normal,requires_grad=True).to(gpu)
            # masked = torch.autograd.Variable(normal,requires_grad=True).to(gpu)

            output = model_cn(masked.to(gpu))
            loss = torch.nn.functional.mse_loss(output, normal.to(gpu))
            
            # backward
            loss.backward()
            # optimize
            opt_cn.step()
            opt_cn.zero_grad()

            if args.wandb!="tmp":
                wandb.log({"phase_1_train_loss": loss.cpu()})
            pbar.set_description('phase 1 | train loss: %.5f' % loss.cpu())
        pbar.update()

        # test
        if pbar.n % args.snaperiod_1 == 0:
            model_cn.eval()
            with torch.no_grad():
                normal, masked = sample_random_batch(
                    test_dset,
                    batch_size=args.num_test_completions)
                normal = normal.to(gpu)
                masked = masked.to(gpu)
                output = model_cn(masked)
                
                # completed = output
                imgs = torch.cat((
                    masked.cpu(),
                    normal.cpu(),
                    output.cpu()), dim=0)
                imgpath = os.path.join(
                    args.result_dir,
                    'phase_1',
                    'step%d.png' % pbar.n)
                model_cn_path = os.path.join(
                    args.result_dir,
                    'phase_1',
                    'model_cn_step%d' % pbar.n)
                save_image(imgs, imgpath, nrow=len(masked))
                
                torch.save(
                    model_cn.state_dict(),
                    model_cn_path)
            model_cn.train()
        if pbar.n >= args.steps_1:
            break
    pbar.close()

    # ================================================
    # Training Phase 2
    # ================================================
    # load context discriminator
    model_cd = ContextDiscriminator(
        local_input_shape=(3, args.ld_input_size, args.ld_input_size),
        global_input_shape=(3, args.cn_input_size, args.cn_input_size))
    if args.init_model_cd is not None:
        model_cd.load_state_dict(torch.load(
            args.init_model_cd,
            map_location='cpu'))
    if args.data_parallel:
        model_cd = DataParallel(model_cd)
    model_cd = model_cd.to(gpu)
    opt_cd = Adadelta(model_cd.parameters())
    bceloss = BCELoss()

    # training
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_2)
    while pbar.n < args.steps_2:
        for i, (normal, masked) in enumerate(train_loader, 0):
            # forward
            normal = normal.to(gpu)
            # print(f"x shape = {x.shape}")
            masked = masked.to(gpu)

            fake = torch.zeros((len(masked), 1)).to(gpu)
            # x_mask = x - x * mask + mpv * mask
            # input_cn = torch.cat((x_mask, mask), dim=1)
            output_cn = model_cn(masked)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake,args.use_one_dis,gpu)
            # print(f"input_gd_fake = {input_gd_fake.shape} input_ld_fake = {input_ld_fake.shape}")
            output_fake = model_cd((
                input_ld_fake.to(gpu),
                input_gd_fake.to(gpu)))
            loss_fake = bceloss(output_fake, fake)

            # real forward
            real = torch.ones((len(masked), 1)).to(gpu)
            input_gd_real = normal
            input_ld_real = crop(input_gd_real,args.use_one_dis,gpu)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_real = bceloss(output_real, real)

            # reduce
            loss = (loss_fake + loss_real) / 2.

            # backward
            loss.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cd.step()
                opt_cd.zero_grad()
                if args.wandb==1:
                    wandb.log({"phase_2_train_loss": loss.cpu()})
                
                pbar.set_description('phase 2 | train loss: %.5f' % loss.cpu())
                pbar.update()

                # test
                if pbar.n % args.snaperiod_2 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        normal, masked = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions)
                        normal = normal.to(gpu)
                        masked = masked.to(gpu)
                        
                        output = model_cn(masked)
                        completed = poisson_blend(masked,output)
                        imgs = torch.cat((
                            masked.cpu(),
                            normal.cpu(),
                            output.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'step%d.png' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_2',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(masked))
                        if args.data_parallel:
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                        else:
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                    model_cn.train()
                if pbar.n >= args.steps_2:
                    break
    pbar.close()

    # ================================================
    # Training Phase 3
    # ================================================
    cnt_bdivs = 0
    pbar = tqdm(total=args.steps_3)
    while pbar.n < args.steps_3:
        for i, (normal, masked) in enumerate(train_loader, 0):
            # forward
            normal = normal.to(gpu)
            masked = masked.to(gpu)

            # fake forward
            fake = torch.zeros((len(masked), 1)).to(gpu)
            output_cn = model_cn(masked)
            input_gd_fake = output_cn.detach()
            input_ld_fake = crop(input_gd_fake,args.use_one_dis,gpu)
            output_fake = model_cd((input_ld_fake, input_gd_fake))
            loss_cd_fake = bceloss(output_fake, fake)

            # real forward
            
            real = torch.ones((len(masked), 1)).to(gpu)
            input_gd_real = normal
            input_ld_real = crop(input_gd_real,args.use_one_dis,gpu)
            output_real = model_cd((input_ld_real, input_gd_real))
            loss_cd_real = bceloss(output_real, real)

            # reduce
            loss_cd = (loss_cd_fake + loss_cd_real) * alpha / 2.

            # backward model_cd
            loss_cd.backward()
            cnt_bdivs += 1
            if cnt_bdivs >= args.bdivs:
                # optimize
                opt_cd.step()
                opt_cd.zero_grad()

            # forward model_cn
            loss_cn_1 = completion_network_loss(output_cn, normal)
            input_gd_fake = output_cn
            input_ld_fake = crop(input_gd_fake,args.use_one_dis,gpu)
            output_fake = model_cd((input_ld_fake, (input_gd_fake)))
            loss_cn_2 = bceloss(output_fake, real)

            # reduce
            loss_cn = (loss_cn_1 + alpha * loss_cn_2) / 2.

            # backward model_cn
            loss_cn.backward()
            if cnt_bdivs >= args.bdivs:
                cnt_bdivs = 0

                # optimize
                opt_cn.step()
                opt_cn.zero_grad()
                if args.wandb==1:
                    wandb.log({"phase_3_train_cd_loss": loss_cd.cpu(),"phase_3_train_cn_loss": loss_cn.cpu() })
                
                pbar.set_description(
                    'phase 3 | train loss (cd): %.5f (cn): %.5f' % (
                        loss_cd.cpu(),
                        loss_cn.cpu()))
                pbar.update()

                # test
                if pbar.n % args.snaperiod_3 == 0:
                    model_cn.eval()
                    with torch.no_grad():
                        normal, masked = sample_random_batch(
                            test_dset,
                            batch_size=args.num_test_completions)
                        normal = normal.to(gpu)
                        masked = masked.to(gpu)

                       
                        output = model_cn(masked)
                        completed = poisson_blend(masked,output)
                        # output = completed
                        imgs = torch.cat((
                            masked.cpu(),
                            normal.cpu(),
                            output.cpu(),
                            completed.cpu()), dim=0)
                        imgpath = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'step%d.png' % pbar.n)
                        model_cn_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cn_step%d' % pbar.n)
                        model_cd_path = os.path.join(
                            args.result_dir,
                            'phase_3',
                            'model_cd_step%d' % pbar.n)
                        save_image(imgs, imgpath, nrow=len(masked))
                        if args.data_parallel:
                            torch.save(
                                model_cn.module.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.module.state_dict(),
                                model_cd_path)
                        else:
                            torch.save(
                                model_cn.state_dict(),
                                model_cn_path)
                            torch.save(
                                model_cd.state_dict(),
                                model_cd_path)
                    model_cn.train()
                if pbar.n >= args.steps_3:
                    break
    pbar.close()
    if args.wandb==1:
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)
    
    main(args)