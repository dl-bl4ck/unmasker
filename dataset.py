from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pathlib import Path
import torch
from tqdm.auto import tqdm

class masked_dataset(Dataset):
    def __init__(self,path="",maxx=0):
        self.normal = []
        self.masked = []
        
        for i in tqdm((Path(path)/'normal').glob("*.jpg")):
            self.normal.append(i)
            name = str(i).split('/')[-1]
            self.masked.append(Path(path)/f"masked/{name}")
            
            if len(self.masked) >= maxx:
                break

            
    def __getitem__(self, idx):
        img = Image.open(self.normal[idx]).resize((224,224))
        x = np.array(img) 
        x = transforms.ToTensor()(img)
 
        mask = Image.open(self.masked[idx]).resize((224,224))
        mask = np.array(mask) 
        mask = transforms.ToTensor()(mask)

        return x,mask

    def __len__(self):
        return len(self.masked)
