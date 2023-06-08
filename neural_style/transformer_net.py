import torch


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Downsampling layers
        self.down1 = DownsampleConvLayer(3, 64, kernel_size=9, downsample=2)
        self.down2 = DownsampleConvLayer(64, 256, kernel_size=3, downsample=2)
        self.down3 = DownsampleConvLayer(256, 512, kernel_size=3, downsample=2)
        #self.down4 = DownsampleConvLayer(128, 256, kernel_size=3, downsample=2)
        #self.down5 = DownsampleConvLayer(256, 512, kernel_size=3, downsample=2)

        # Residual layers
        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        self.res4 = ResidualBlock(512)
        self.res5 = ResidualBlock(512)
        self.res6 = ResidualBlock(512)
        #self.res7 = ResidualBlock(512)

        # Upsampling Layers
        

        #withot skips
        self.up1 = UpsampleConvLayer(512, 256, kernel_size=3, stride=1, upsample=2)
        self.up2 = UpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)
        self.up3 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        #self.up4 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        #self.up5 = UpsampleConvLayer(32, 32, kernel_size=3, stride=1, upsample=2)
        
        #with skips
        #self.up1 = UpsampleConvLayer(512 + 32, 256, kernel_size=3, stride=1, upsample=2)
        #self.up2 = UpsampleConvLayer(256 + 2, 128, kernel_size=3, stride=1, upsample=2)
        #self.up3 = UpsampleConvLayer(128 + 1, 64, kernel_size=3, stride=1, upsample=2)
        # self.up4 = UpsampleConvLayer(64 + 32, 32, kernel_size=3, stride=1, upsample=2)
        # self.up5 = UpsampleConvLayer(32 + 16, 32, kernel_size=3, stride=1, upsample=2)
        self.final = ConvLayer(32, 3, kernel_size=9, stride=1)

        # Skips 
        #self.skip5 = ConvLayer(512, 32, 3, 1)
        #self.skip4 = ConvLayer(256, 2, 3, 1)
        #self.skip3 = ConvLayer(128, 1, 3, 1)
        # self.skip2 = ConvLayer(64, 32, 3, 1)
        # self.skip1 = ConvLayer(32, 16, 3, 1)

    def forward(self, X):
        y = X
        y = self.down1(y)
        # s1 = self.skip1(y)

        y = self.down2(y)
        # s2 = self.skip2(y)

        y = self.down3(y)
        #s3 = self.skip3(y)

        #y = self.down4(y)
        #s4 = self.skip4(y)

        #y = self.down5(y)
        #s5 = self.skip5(y)

        # --------------------------------

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        #y = self.res7(y)

        # --------------------------------

        #y = torch.cat((y, s5), 1)
        y = self.up1(y)

        #y = torch.cat((y, s4), 1)
        y = self.up2(y)

        #y = torch.cat((y, s3), 1)
        y = self.up3(y)

        # y = torch.cat((y, s2), 1)
        #y = self.up4(y)

        # y = torch.cat((y, s1), 1)
        #y = self.up5(y)

        y = self.final(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.norm1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.norm2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.act = torch.nn.GELU()

    def forward(self, x):
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = x + residual
        return x



class DownsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=2):
        super(DownsampleConvLayer, self).__init__()
        self.downsample = downsample
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, downsample)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1)
        self.norm1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.norm2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.conv2 = ConvLayer(out_channels, out_channels, 3, stride)
        self.norm1 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.norm2 = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.act = torch.nn.GELU()

    def forward(self, x):
        if self.upsample:
            x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=self.upsample)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x
