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
            # nn.Dropout2d(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            # nn.Dropout2d(0.25),
        )
        
        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
        )
        
        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv11 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv12 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
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

if __name__ == "__main__":
    
    model = CompletionNetwork(debug=True)
    preds = model(torch.Tensor(1,3,224,224))
    print(preds.shape)


