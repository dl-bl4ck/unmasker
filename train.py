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
parser.add_argument('--learning_rate',type=float,default=0.0001)


        
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
    opt_cn = Adam(model_cn.parameters(),lr=args.learning_rate)

    # training
    # ================================================
    # Training Phase 1
    # ================================================
    pbar = tqdm(total=args.steps_1)
    
    epochs = 0
    while epochs < args.steps_1:
        for i, (normal, masked) in tqdm(enumerate(train_loader, 0)):
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
        epochs += 1
    pbar.close()
    


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    if args.init_model_cd is not None:
        args.init_model_cd = os.path.expanduser(args.init_model_cd)
    
    main(args)
