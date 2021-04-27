import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Flatten, Concatenate


class CompletionNetwork(nn.Module):
    def __init__(self,debug = False):
        super(CompletionNetwork, self).__init__()
        self.debug = debug
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv12 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

    def forward(self, x):
        if self.debug:
            print(f"Input = {x.shape}")
        x1 = self.conv1(x)

        if self.debug:
            print(f"Layer1 = {x1.shape}")
        
        x2 = self.conv2(x1)
        
        if self.debug:
            print(f"Layer2 = {x2.shape}")
        x3 = self.conv3(x2)
        
        if self.debug:
            print(f"Layer3 = {x3.shape}")

        x4 = self.conv4(x3)
        if self.debug:
            print(f"Layer4 = {x4.shape}")

        x5 = self.conv5(x4)
        if self.debug:

            print(f"Layer5 = {x5.shape}")

        x6 = self.conv6(x5)
        if self.debug:

            print(f"Layer6 = {x6.shape}")
        x7 = self.conv7(x6)
        if self.debug:

            print(f"Layer7 = {x7.shape}")
        x8 = self.conv8(x7)
        if self.debug:

            print(f"Layer8 = {x8.shape}")
        
        x8_input = torch.cat((x8,x5),1)
        x9 = self.conv9(x8_input)
        if self.debug:

            print(f"Layer9 = {x9.shape}")
        x9_input = torch.cat((x9,x4),1)
        x10 = self.conv10(x9_input)
        if self.debug:

            print(f"Layer10 = {x10.shape}")

        x10_input = torch.cat((x10,x3),1)
        x11 = self.conv11(x10_input)
        if self.debug:

            print(f"Layer11 = {x11.shape}")
        x11_input = torch.cat((x11,x2),1)
        x12 = self.conv12(x11_input)
        if self.debug:

            print(f"Layer12 = {x12.shape}")

        x12_input = torch.cat((x12,x1),1)
        x13 = self.conv13(x12_input)
        if self.debug:

            print(f"Layer13 = {x13.shape}")

        return x13

class LocalDiscriminator(nn.Module):
    def __init__(self, input_shape = (3,96,96), output_neurons=256,debug = False):
        super(LocalDiscriminator, self).__init__()
        self.debug = debug
        self.input_shape = input_shape
        self.output_shape = (256,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.img_c, 64, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 256* 6 *6
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
        )


    def forward(self, x):
        
        if self.debug:
            print(f"Input = {x.shape}")
        x = self.conv1(x)
        if self.debug:
            print(f"LAYER 1 = {x.shape}")
        
        x = self.conv2(x)
        if self.debug:
            print(f"LAYER 2 = {x.shape}")
        x = self.conv3(x)
        if self.debug:
            print(f"LAYER 3 = {x.shape}")
        x = self.conv4(x)
        if self.debug:
            print(f"LAYER 4 = {x.shape}")
    
        x = self.linear(x)
        if self.debug:
            print(f"output = {x.shape}")
        return x


class GlobalDiscriminator(nn.Module):
    def __init__(self, input_shape = (3,224,244), debug = False):
        super(GlobalDiscriminator, self).__init__()
        self.debug = debug
        self.input_shape = input_shape
        self.output_shape = (1024,)
        self.img_c = input_shape[0]
        self.img_h = input_shape[1]
        self.img_w = input_shape[2]

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.img_c, 64, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(0.25),
        )
        
        # input_shape: (None, 512, img_h//32, img_w//32)
        in_features = 1024* 4*4
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
        )


    def forward(self, x):
        if self.debug:
            print(f"Input = {x.shape}")
        x = self.conv1(x)
        if self.debug:
            print(f"LAYER 1 = {x.shape}")
        
        x = self.conv2(x)
        if self.debug:
            print(f"LAYER 2 = {x.shape}")
        x = self.conv3(x)
        if self.debug:
            print(f"LAYER 3 = {x.shape}")
        x = self.conv4(x)
        if self.debug:
            print(f"LAYER 4 = {x.shape}")

        x = self.conv5(x)
        if self.debug:
            print(f"LAYER 5 = {x.shape}")
    
        x = self.conv6(x)
        if self.debug:
            print(f"LAYER 6 = {x.shape}")
    
        x = self.linear(x)
        if self.debug:
            print(f"output = {x.shape}")
        return x


class ContextDiscriminator(nn.Module):
    def __init__(self,debug = False):
        super(ContextDiscriminator, self).__init__()
        self.debug = debug
        self.model_ld = LocalDiscriminator(debug=self.debug)
        self.model_gd = GlobalDiscriminator(debug=self.debug)

        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        
        self.concat = Concatenate(dim=-1)
        
        self.linear = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x_ld, x_gd):
        if self.debug:
            print(f"Input : x_ld = {x_ld.shape} x_gd = {x_gd.shape}")
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        if self.debug:
            print(f"Output : x_ld = {x_ld.shape} x_gd = {x_gd.shape}")

        x = self.linear(self.concat([x_ld, x_gd]))

        return x

if __name__ == "__main__":
    
    # for k in range(2,4):
    #     for s in range(1,4):
    #         for p in range(4):
    #             layer = nn.ConvTranspose2d(512, 512, kernel_size=k, stride=s, padding = p)(torch.Tensor(1,512,14,14))
    #             if layer.shape[2] == 28 and layer.shape[3] == 28:
    #                 print(k,s,p)
    x_gd = torch.randn(1,3,224,224)
    x_ld = torch.randn(1,3,96,96)
    model = ContextDiscriminator(debug=True)
    preds = model(x_gd=x_gd,x_ld=x_ld)
    print(preds.shape)


