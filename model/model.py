import torch
import torch.nn as nn
import torch.nn.functional as F
from model.convlstm import ConvLSTMCells3D

def load_CRNN(dir,device,net):
    state_dict = torch.load(dir, map_location=device)
    if 'lstm1.cell_list.0.h_cur' in state_dict:
        net.init_lstm_states(list(state_dict['lstm1.cell_list.0.h_cur'].shape))
    if 'lstm1.cell_list.0.c_cur' in state_dict:
        net.init_lstm_states(list(state_dict['lstm1.cell_list.0.c_cur'].shape))
    net.load_state_dict(state_dict,strict=False)
    return


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv3d(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3DLSTM(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=64, bilinear=True):
        # example input_shape: [N, C, D, H, W]
        super(UNet3DLSTM, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        self.lstm1 = ConvLSTMCells3D(inChannel=fChannel,hChannel=fChannel,kernel_size=3,num_layers=1)
        self.lstm2 = ConvLSTMCells3D(inChannel=fChannel*2,hChannel=fChannel*2,kernel_size=3,num_layers=1)
        self.lstm3 = ConvLSTMCells3D(inChannel=fChannel*4,hChannel=fChannel*4,kernel_size=3,num_layers=1)
        self.lstm4 = ConvLSTMCells3D(inChannel=fChannel*8,hChannel=fChannel*8,kernel_size=3,num_layers=1)
        self.lstm5 = ConvLSTMCells3D(inChannel=fChannel*16//factor,hChannel=fChannel*16//factor,kernel_size=3,num_layers=1)

        self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
        self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
        self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
        self.up4 = Up3D(fChannel*2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, outChannel)

    def reset_lstm(self):
        self.lstm1.reset()
        self.lstm2.reset()
        self.lstm3.reset()
        self.lstm4.reset()
        self.lstm5.reset()
        return

    def init_lstm_states(self,input_size):
        return
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1 = self.lstm1(x1)
        x2 = self.lstm2(x2)
        x3 = self.lstm3(x3)
        x4 = self.lstm4(x4)
        x5 = self.lstm5(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
      

class UNet3D(nn.Module):
    def __init__(self, inChannel, outChannel, fChannel=64, bilinear=True):
        super(UNet3D, self).__init__()
        self.inChannel = inChannel
        self.outChannel = outChannel
        self.fChannel=fChannel
        self.bilinear = bilinear

        self.inc = DoubleConv3D(inChannel, fChannel)
        self.down1 = Down3D(fChannel, fChannel*2)
        self.down2 = Down3D(fChannel*2, fChannel*4)
        self.down3 = Down3D(fChannel*4, fChannel*8)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(fChannel*8, fChannel*16 // factor)

        self.up1 = Up3D(fChannel*16, fChannel*8 // factor, bilinear)
        self.up2 = Up3D(fChannel*8, fChannel*4 // factor, bilinear)
        self.up3 = Up3D(fChannel*4, fChannel*2 // factor, bilinear)
        self.up4 = Up3D(fChannel*2, fChannel, bilinear)
        self.outc = OutConv3D(fChannel, outChannel)

    def reset_lstm(self):
        return
    def init_lstm_states(self,input_size):
        return
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    model=UNet3D(2,2)
    input=torch.randn(size=[1,2,64,64,64])
    output1=model(input)
    output2=model(input)
    model.reset_lstm()
    pass
