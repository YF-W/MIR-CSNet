import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),)


    def forward(self, x):
        return self.conv(x)

class NetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, r = 16):
        super(NetBlock, self).__init__()

        #GMP,GAP
        self.GMP = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channel, out_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channel, out_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Sigmoid(),
        )

        self.conv2_01 = ConvBlock(in_channel, out_channel)
        self.conv2_02 = ConvBlock(in_channel, out_channel)
        self.conv2_03 = ConvBlock(in_channel, out_channel)
        self.conv2_04 = ConvBlock(in_channel, out_channel)
        self.conv2_1_01 = ConvBlock(out_channel, out_channel)
        self.conv2_1_02 = ConvBlock(out_channel, out_channel)
        self.conv2_1_03 = ConvBlock(out_channel, out_channel)
        self.conv2_1_04 = ConvBlock(out_channel, out_channel)
        self.conv2_2 = ConvBlock(out_channel*2, out_channel)

        self.conv_r2_01 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=2, dilation=2)
        self.conv_r2_02 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=2, dilation=2)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        h,w = x.size()[2:]
        h_1, w_1=h,w
        h = int(round(h / 4 * 3))
        w = int(round(w / 4 * 3))

        x1_1 = x[:, :, :h, :w]
        x1_2 = x[:, :, :h, -w:]
        x2_1 = x[:, :, -h:, :w]
        x2_2 = x[:, :, -h:, -w:]

        x1_1 = self.conv2_01(x1_1)
        x1_2 = self.conv2_02(x1_2)
        x2_1 = self.conv2_03(x2_1)
        x2_2 = self.conv2_04(x2_2)

        x1_1M = self.GMP(x1_1)
        x1_1A = self.GAP(x1_1)
        x1_1M = x1_1M.unsqueeze(-1).unsqueeze(-1).expand_as(x1_1)
        x1_1A = x1_1A.unsqueeze(-1).unsqueeze(-1).expand_as(x1_1)
        x1_1M = x1_1*x1_1M
        x1_1A = x1_1*x1_1A

        x1_2M = self.GMP(x1_2)
        x1_2A = self.GAP(x1_2)
        x1_2M = x1_2M.unsqueeze(-1).unsqueeze(-1).expand_as(x1_2)
        x1_2A = x1_2A.unsqueeze(-1).unsqueeze(-1).expand_as(x1_2)
        x1_2M = x1_2*x1_2M
        x1_2A = x1_2*x1_2A

        x2_1M = self.GMP(x2_1)
        x2_1A = self.GAP(x2_1)
        x2_1M = x2_1M.unsqueeze(-1).unsqueeze(-1).expand_as(x2_1)
        x2_1A = x2_1A.unsqueeze(-1).unsqueeze(-1).expand_as(x2_1)
        x2_1M = x2_1*x2_1M
        x2_1A = x2_1*x2_1A

        x2_2M = self.GMP(x2_2)
        x2_2A = self.GAP(x2_2)
        x2_2M = x2_2M.unsqueeze(-1).unsqueeze(-1).expand_as(x2_2)
        x2_2A = x2_2A.unsqueeze(-1).unsqueeze(-1).expand_as(x2_2)
        x2_2M = x2_2*x2_2M
        x2_2A = x2_2*x2_2A

        x1_1M = F.pad(x1_1M, (0, w_1//4, 0, 0), mode='constant', value=0)
        x1_2M = F.pad(x1_2M, (w_1//4, 0, 0, 0), mode='constant', value=0)
        x_up = x1_1M + x1_2M

        x2_1M = F.pad(x2_1M, (0, w_1//4, 0, 0), mode='constant', value=0)
        x2_2M = F.pad(x2_2M, (w_1//4, 0, 0, 0), mode='constant', value=0)
        x_down = x2_1M + x2_2M

        x1_1A = F.pad(x1_1A, (0, 0, 0, h_1//4), mode='constant', value=0)
        x2_1A = F.pad(x2_1A, (0, 0, h_1//4, 0), mode='constant', value=0)
        x_left = x1_1A + x2_1A

        x1_2A = F.pad(x1_2A, (0, 0, 0, h_1//4), mode='constant', value=0)
        x2_2A = F.pad(x2_2A, (0, 0, h_1//4, 0), mode='constant', value=0)
        x_right = x1_2A + x2_2A

        x_up = self.conv2_1_01(x_up)
        x_down = self.conv2_1_02(x_down)
        x_left = self.conv2_1_03(x_left)
        x_right = self.conv2_1_04(x_right)

        x_up = F.pad(x_up, (0, 0, 0, h_1//4), mode='constant', value=0)
        x_down = F.pad(x_down, (0, 0, h_1//4, 0), mode='constant', value=0)
        x1 = x_up + x_down
        x1 = self.conv_r2_01(x1)

        x_left = F.pad(x_left, (0, w_1//4, 0, 0), mode='constant', value=0)
        x_right = F.pad(x_right, (w_1//4, 0, 0, 0), mode='constant', value=0)
        x2 = x_left + x_right
        x2 = self.conv_r2_02(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.conv2_2(x)
        return x


class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = torch.reshape(x, (batch_size, self.num_groups, chs_per_group, h, w))
        x = x.transpose(1, 2)
        out = torch.reshape(x, (batch_size, -1, h, w))
        return out

class DoubleConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv5, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv7(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(

        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class CNet(nn.Module):
    def __init__(
            self
    ):
        super(CNet, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_1 = nn.Conv2d(3,64,1)
        self.conv1_2 = nn.Conv2d(128, 64, 1)
        self.conv1_3 = nn.Conv2d(256, 128, 1)
        self.conv1_4 = nn.Conv2d(512, 256, 1)

        self.conv1 = nn.Conv2d(64, 1, 1)

        self.dbconv1_1 = DoubleConv5(1, 1)
        self.dbconv1_2 = DoubleConv3(1, 1)
        self.dbconv1_3 = DoubleConv7(1, 1)
        self.dbconv1_4 = DoubleConv3(3, 64)
        self.dbconv2_1 = DoubleConv5(16, 32)
        self.dbconv2_2 = DoubleConv7(16, 32)
        self.dbconv2_3 = DoubleConv3(64, 128)
        self.dbconv3_1 = DoubleConv5(32, 64)
        self.dbconv3_2 = DoubleConv7(32, 64)
        self.dbconv3_3 = DoubleConv3(128, 256)
        self.dbconv4_1 = DoubleConv5(64, 128)
        self.dbconv4_2 = DoubleConv7(64, 128)
        self.dbconv4_3 = DoubleConv3(256, 512)

        self.dbconv5_1 = DoubleConv3(256, 512)
        self.dbconv5_2 = DoubleConv3(1024, 512)
        self.dbconv6_1 = DoubleConv3(128, 256)
        self.dbconv6_2 = DoubleConv3(512, 256)
        self.dbconv7_1 = DoubleConv3(64, 128)
        self.dbconv7_2 = DoubleConv3(256, 128)
        self.dbconv8_1 = DoubleConv3(32, 64)
        self.dbconv8_2 = DoubleConv3(128, 64)

        self.dbconv9 = DoubleConv3(512, 512)
        self.dbconv10 = DoubleConv3(512, 512)

        self.dwconv2 = DepthwiseSeparableConv(32,64)
        self.dwconv3 = DepthwiseSeparableConv(64, 128)
        self.dwconv4 = DepthwiseSeparableConv(128, 256)

        self.dwconv5 = DepthwiseSeparableConv(512, 512)
        self.dwconv6 = DepthwiseSeparableConv(512, 256)
        self.dwconv7 = DepthwiseSeparableConv(256, 128)
        self.dwconv8 = DepthwiseSeparableConv(128, 64)

        self.shuffle3 = Channel_Shuffle(3)
        self.shuffle4 = Channel_Shuffle(4)

        self.netBlock_1 = NetBlock(64, 64)
        self.netBlock_2 = NetBlock(128, 128)
        self.netBlock_3 = NetBlock(256, 256)

    def forward(self, x):
        x_1, x_2, x_3 = torch.split(x, 1, dim=1)
        x_1 = self.dbconv1_1(x_1)
        x_2 = self.dbconv1_2(x_2)
        x_3 = self.dbconv1_3(x_3)

        x_res = torch.cat([x_1, x_2, x_3], dim=1)
        x = x + x_res
        x = self.shuffle3(x)
        x_skip1 = self.dbconv1_4(x)

        x = self.pool(x_skip1)

        x_1, x_2, x_3 = torch.split(x, [16, 32, 16], dim=1)
        x_1 = self.dbconv2_1(x_1)
        x_2 = self.dwconv2(x_2)
        x_3 = self.dbconv2_2(x_3)

        x_res = torch.cat([x_1, x_2, x_3], dim=1)
        x_res = self.conv1_2(x_res)
        x = x + x_res
        x = self.shuffle4(x)
        x_skip2 = self.dbconv2_3(x)

        x = self.pool(x_skip2)

        x_1, x_2, x_3 = torch.split(x, [32, 64, 32], dim=1)
        x_1 = self.dbconv3_1(x_1)
        x_2 = self.dwconv3(x_2)
        x_3 = self.dbconv3_2(x_3)

        x_res = torch.cat([x_1, x_2, x_3], dim=1)
        x_res = self.conv1_3(x_res)
        x = x + x_res
        x = self.shuffle4(x)
        x_skip3 = self.dbconv3_3(x)

        x = self.pool(x_skip3)

        x_1, x_2, x_3 = torch.split(x, [64, 128, 64], dim=1)
        x_1 = self.dbconv4_1(x_1)
        x_2 = self.dwconv4(x_2)
        x_3 = self.dbconv4_2(x_3)

        x_res = torch.cat([x_1, x_2, x_3], dim=1)
        x_res = self.conv1_4(x_res)
        x = x + x_res
        x = self.shuffle4(x)
        x_skip4 = self.dbconv4_3(x)

        x = self.pool(x_skip4) # 512 14

        x1 = self.dbconv9(x)
        x2 = x1 + x
        x3 = self.dbconv10(x2)
        x = x3 + x2

        x_skip1 = self.netBlock_1(x_skip1)
        x_skip2 = self.netBlock_2(x_skip2)
        x_skip3 = self.netBlock_3(x_skip3)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dwconv5(x)
        x_1, x_2 = torch.split(x, [256, 256], dim=1)
        x_1 = self.dbconv5_1(x_1)
        x_2 = self.dbconv5_1(x_2)
        x = x_1 + x_2
        x = torch.cat([x, x_skip4], dim=1)
        x = self.dbconv5_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dwconv6(x)
        x_1, x_2 = torch.split(x, [128, 128], dim=1)
        x_1 = self.dbconv6_1(x_1)
        x_2 = self.dbconv6_1(x_2)
        x = x_1 + x_2
        x = torch.cat([x, x_skip3], dim=1)
        x = self.dbconv6_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dwconv7(x)
        x_1, x_2 = torch.split(x, [64, 64], dim=1)
        x_1 = self.dbconv7_1(x_1)
        x_2 = self.dbconv7_1(x_2)
        x = x_1 + x_2
        x = torch.cat([x, x_skip2], dim=1)
        x = self.dbconv7_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.dwconv8(x)
        x_1, x_2 = torch.split(x, [32, 32], dim=1)
        x_1 = self.dbconv8_1(x_1)
        x_2 = self.dbconv8_1(x_2)
        x = x_1 + x_2
        x = torch.cat([x, x_skip1], dim=1)
        x = self.dbconv8_2(x)

        x = self.conv1(x)

        return x

if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    model = CNet()
    preds = model(x)
    print(x.shape)
    print(preds.shape)

