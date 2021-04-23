import json
import os
from pathlib import Path
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adadelta

from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
from tqdm import tqdm
from models import CompletionNetwork 
from losses import completion_network_loss
import wandb
import pickle
from utils import (
    sample_random_batch,
    poisson_blend,
)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',type=str,default="./dataset/")
parser.add_argument('--wandb_project',type=str,default="tmp")
parser.add_argument('--max_train',type=int,default=32000)
parser.add_argument('--max_test',type=int,default=3200)
parser.add_argument('--result_dir',type=str, default="./results/")
parser.add_argument('--init_model_cn', type=str, default=None)
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--snaperiod', type=int, default=100)
parser.add_argument('--alpha', type=float, default=4e-4)
parser.add_argument('--num_test_completions',type=int,default=32)
parser.add_argument('--batch_size',type=int,default=64)


class my_loader(Dataset):
    def __init__(self,path="",maxx=0):
        self.normal = []
        self.masked = []
        
        for i in tqdm((Path(path)/'normal').glob("*.jpg")):
            img = Image.open(i)
            x = np.array(img) 
            name = str(i).split('/')[-1]
            x = transforms.ToTensor()(img)
            self.normal.append(x)

            mask = Image.open(Path(path)/f"masked/{name}")
            mask = np.array(mask) 
            self.masked.append(transforms.ToTensor()(mask))

            if len(self.masked) >= maxx:
                break

        self.normal = torch.stack(self.normal ,dim = 0)
        self.masked = torch.stack(self.masked ,dim = 0)
            
    def __getitem__(self, idx):
        return self.normal[idx],self.masked[idx]

    def __len__(self):
        return len(self.masked)
        
def main(args):
    if args.wandb_project!="tmp":
        wandb.init(project=args.wandb_project, config=args)

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
    train_dset = my_loader(os.path.join(args.data_dir, 'train'), args.max_train)

    test_dset = my_loader(os.path.join(args.data_dir, 'test'), args.max_test)
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
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
    opt_cn = Adadelta(model_cn.parameters())

    # training
    pbar = tqdm(total=args.epochs)
    while pbar.n < args.epochs:
        for i, (normal, masked) in enumerate(train_loader, 0):
            normal = normal.to(gpu)
            masked = masked.to(gpu)
            
            output = model_cn(masked)
            loss = completion_network_loss(output, normal)

            loss.backward()
            
            opt_cn.step()
            opt_cn.zero_grad()
            if args.wandb_project!="tmp":
                wandb.log({"phase_1_train_loss": loss.cpu()})
            pbar.set_description('train loss: %.5f' % loss.cpu())
            pbar.update()
            if pbar.n % args.snaperiod == 0:
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
                    imgpath = os.path.join(args.result_dir,'step%d.png' % pbar.n)
                    model_cn_path = os.path.join(args.result_dir,'phase_1','model_cn_step%d' % pbar.n)
                    save_image(imgs, imgpath, nrow=len(masked))
                    torch.save(model_cn.state_dict(),model_cn_path)
                model_cn.train()
            if pbar.n >= args.epochs:
                break
    pbar.close()


    if args.wandb_project!="tmp":
        wandb.finish()


if __name__ == '__main__':
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    args.result_dir = os.path.expanduser(args.result_dir)
    if args.init_model_cn is not None:
        args.init_model_cn = os.path.expanduser(args.init_model_cn)
    
    
    main(args)
