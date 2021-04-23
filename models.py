import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Flatten, Concatenate


class CompletionNetwork(nn.Module):
    def __init__(self):
        super(CompletionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.act4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.act5 = nn.ReLU()
        
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, dilation = 1)
        self.bn6 = nn.BatchNorm2d(32)
        self.act6 = nn.ReLU()
        
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, dilation=1, padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.act7 = nn.ReLU()
        
        self.conv8 = nn.Conv2d(16, 3, kernel_size=3, stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(3)
        self.act8 = nn.ReLU()

    def forward(self, x):
        # print(f"Input = {x.shape}")
        x = self.bn1(self.act1(self.conv1(x)))
        # print(f"Layer1 = {x.shape}")
        x = self.bn2(self.act2(self.conv2(x)))
        # print(f"Layer2 = {x.shape}")
        x = self.bn3(self.act3(self.conv3(x)))
        # print(f"Layer3 = {x.shape}")
        x = self.bn4(self.act4(self.conv4(x)))
        # print(f"Layer4 = {x.shape}")
        x = self.bn5(self.act5(self.conv5(x)))
        # print(f"Layer5 = {x.shape}")
        x = self.bn6(self.act6(self.conv6(x)))
        # print(f"Layer6 = {x.shape}")
        x = self.bn7(self.act7(self.conv7(x)))
        # print(f"Layer7 = {x.shape}")
        x = self.bn8(self.act8(self.conv8(x)))
        # print(f"Layer8 = {x.shape}")
        return x


class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(LocalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (256,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        self.conv1 = nn.Conv2d(self.img_c, 16, kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(16, 32,  kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 32*6*7
        self.flatten3 = Flatten()
        self.linear3 = nn.Linear(in_features, 256)
        self.act3 = nn.ReLU()

    def forward(self, x):
        # print(f"haha Input = {x.shape}")
        x = self.bn1(self.act1(self.conv1(x)))
        # print(f"layer1 = {x.shape}")

        x = self.bn2(self.act2(self.conv2(x)))

        # print(f"layer2 = {x.shape}")

        x = self.act3(self.linear3(self.flatten3(x)))
        # print(f"layer3 = {x.shape}")

        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(GlobalDiscriminator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (256,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        # input_shape: (None, img_c, img_h, img_w)
        self.conv1 = nn.Conv2d(self.img_c, 16, kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.ReLU()
        # input_shape: (None, 64, img_h//2, img_w//2)
        self.conv2 = nn.Conv2d(16, 32,  kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.act2 = nn.ReLU()
        # input_shape: (None, 128, img_h//4, img_w//4)
        self.conv3 = nn.Conv2d(32, 64,  kernel_size=3, stride=2, padding=1, dilation = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.act3 = nn.ReLU()
        
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 64*8*8
        self.flatten4 = Flatten()
        self.linear4 = nn.Linear(in_features, 256)
        self.act4 = nn.ReLU()


    def forward(self, x):
        # print(f"haha Input = {x.shape}")

        x = self.bn1(self.act1(self.conv1(x)))
        # print(f"layer1 = {x.shape}")
        
        x = self.bn2(self.act2(self.conv2(x)))
        # print(f"layer2 = {x.shape}")
        
        x = self.bn3(self.act3(self.conv3(x)))
        # print(f"layer3 = {x.shape}")
        
        x = self.act4(self.linear4(self.flatten4(x)))
        # print(f"layer4 = {x.shape}")
        
        return x


class ContextDiscriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape):
        super(ContextDiscriminator, self).__init__()
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape)
        # input_shape: [(None, 1024), (None, 1024)]
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        # print(f"in_features = {in_features}")
        self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()
        # output_shape: (None, 1)

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        # print(f"x_ld shape = {x_ld.shape} x_gd shape = {x_gd.shape}")
        out = self.act1(self.linear1(self.concat1([x_ld, x_gd])))
        return out
