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
            img = Image.open(i).resize((224,224))
            x = np.array(img) 

            name = str(i).split('/')[-1]
            x = transforms.ToTensor()(img)
            self.normal.append(x)     

            mask = Image.open(Path(path)/f"masked/{name}").resize((224,224))
            mask = np.array(mask) 
            self.masked.append(transforms.ToTensor()(mask))

            if len(self.masked) >= maxx:
                break

        self.normal = torch.stack(self.normal ,dim = 0)
        self.masked = torch.stack(self.masked ,dim = 0)
        #print(self.normal.shape,self.masked.shape)
            
    def __getitem__(self, idx):
        return self.normal[idx],self.masked[idx]

    def __len__(self):
        return len(self.masked)
