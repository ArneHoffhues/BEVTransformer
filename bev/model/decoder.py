import torch
import torch.nn as nn
import torch.nn.functional as F

from bev.model.resnet import ResNetLayer
#from resnet import ResNetLayer

class UpsampleNetwork(nn.Sequential):

    def __init__(self, in_channels, layers=[6, 1, 1, 1], 
                 strides=[1, 2, 2, 2], blocktype='basic'):
        
        modules = list()
        self.downsample = 1
        channels = in_channels
        for nblocks, stride in zip(layers, strides):

            # Add a new residual layer
            module = ResNetLayer(channels, 
                    channels // 2, nblocks, 1/stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            channels = channels // 2
            self.downsample *= stride
        
        self.out_channels = channels

        
        super().__init__(*modules)

class Decoder(nn.Module):

    def __init__(self,in_channels, num_classes, predict_vis = False, layers=[6, 1, 1, 1],
            strides=[1, 2, 2, 2], blocktype='basic', final_res=None):
        super().__init__()
        self.predict_vis = predict_vis
        self.final_res = tuple(final_res)

        self.upsample = UpsampleNetwork(in_channels, layers,
                strides, blocktype)
        
        if final_res is not None:
            self.conv_after_interpolation = torch.nn.Conv2d(self.upsample.out_channels, self.upsample.out_channels, 
                    kernel_size = 3, stride = 1, padding = 1)

        self.conv_final = nn.Conv2d(self.upsample.out_channels, num_classes, 
                kernel_size = 1, stride = 1)
        
        if predict_vis:
            self.conv_vis = nn.Conv2d(self.upsample.out_channels, 1, kernel_size = 1, stride = 1)

    def forward (self, x):
        x = self.upsample(x)
        
        if self.final_res is not None:
            x = F.interpolate(x, size=self.final_res, mode='nearest')
            x = self.conv_after_interpolation(x)

        if self.predict_vis:
            return self.conv_final(x), self.conv_vis(x)
        else:
            return self.conv_final(x), None

class SegAndDensityDecoder(nn.Module):

    def __init__(self, in_channels, num_classes, num_upsample_before_concat = 2, num_upsample_after_concat = 1, 
            final_res = (676, 676), predict_vis = False):
        super().__init__()
        
        self.predict_vis = predict_vis
        self.final_res = tuple(final_res)

        modules = list()
        for _ in range(num_upsample_before_concat):
            module = ResNetLayer(in_channels, in_channels, 1, 1/2, blocktype='basic')
            modules.append(module)
        self.upsample_seg = nn.Sequential(*modules)
        
        channels = in_channels * 2
        modules = list()
        for _ in range(2):
            module = ResNetLayer(channels, channels // 2, 3, 1, blocktype='basic')
            modules.append(module)
            channels = channels // 2
        self.concat_network = nn.Sequential(*modules)

        modules = list()
        for _ in range(num_upsample_after_concat):
            module = ResNetLayer(channels, channels // 2, 1, 1/2, blocktype='basic')
            modules.append(module)
            channels = channels // 2
        self.upsample_after_concat = nn.Sequential(*modules)
        
        self.conv_after_interpolation = torch.nn.Conv2d(channels, channels // 2, kernel_size = 3, stride = 1, padding = 1)
        channels = channels // 2

        self.conv_final = nn.Conv2d(channels, num_classes,
                kernel_size = 1, stride = 1)

        if predict_vis:
            self.conv_vis = nn.Conv2d(channels, 1, kernel_size = 1, stride = 1)

    def forward(self, seg, density):
        seg = self.upsample_seg(seg)
        cat = torch.cat([seg, density], dim = 1)
        out = self.concat_network(cat)
        out = self.upsample_after_concat(out)
        out = F.interpolate(out, size=self.final_res, mode='nearest')
        out = self.conv_after_interpolation(out)

        if self.predict_vis:
            return self.conv_final(out), self.conv_vis(out)
        else:
            return self.conv_final(out), None

def _test():
    decoder = SegAndDensityDecoder(256, 10, predict_vis = True).cuda()
    for _ in range(10000):
        seg = torch.rand((3, 256, 50, 50), dtype = torch.float32).cuda()
        density = torch.rand((3, 256, 200, 200), dtype = torch.float32).cuda()
        out, vis = decoder(seg, density)
        print(out.shape)
        print(vis.shape)

if __name__ == '__main__':
    _test()

