import os
import argparse
import json
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
from PIL import Image
from models import CompletionNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--input_img')
parser.add_argument('--output_img',type=str,default="output.jpg")

def predict(model_path, input_img):
    

    model = CompletionNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    img = input_img.resize((224,224))
    x = transforms.ToTensor()(img)
    x = torch.unsqueeze(x, 0)
    # print(x.shape)

    model.eval()
    with torch.no_grad():
        output = model(x)
        # save_image(output, args.output_img, nrow=3)
    # print('output img was saved as %s.' % args.output_img)
    return transforms.ToPILImage()(output[0]).convert("RGB")

if __name__ == '__main__':
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model)
    input_img = Image.open(os.path.expanduser(args.input_img))
    output = predict(model_path,input_img).save('./tmp2.jpg')