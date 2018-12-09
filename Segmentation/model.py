import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
       )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownSampleBlock, self).__init__()
        self.mp = nn.Sequential(
            nn.MaxPool2d(2, 2),
            double_conv(in_c, out_c)
        )

    def forward(self, x):
        x = self.mp(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpSampleBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_c//2, in_c//2, 2, stride=2)
        self.conv = double_conv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv0 = double_conv(3, 64)
        self.down1 = DownSampleBlock(64, 128)
        self.down2 = DownSampleBlock(128, 256)
        self.down3 = DownSampleBlock(256, 512)
        self.down4 = DownSampleBlock(512, 512)
        
        self.up1 = UpSampleBlock(1024, 256)
        self.up2 = UpSampleBlock(512, 128)
        self.up3 = UpSampleBlock(256, 64)
        self.up4 = UpSampleBlock(128, 64)
        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.final_conv(x)
        return x
    
    def predict(self, x):
        out = self.forward(x.unsqueeze(0).cuda())
        out = out > 0
        out = out.squeeze(0).squeeze(0).float().cuda()
        return out
