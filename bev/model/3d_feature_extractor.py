import torch
from torch import nn
import torch.nn.functional as F

from bev.model.utils import construct_volume_lattice
from bev.model.resnet import ResNetLayer
from bev.model.unet.unet_model import UNet3D

class RayProjectionLayer(nn.Module):

    def __init__(self, scales = [4, 8, 16, 32, 64], in_ch= 256, out_ch = 128, bev_grid_size=(49, 50), 
            grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=50):
        super().__init__()
        self.scales = scales

        self.register_buffer('volume_grid', construct_volume_lattice(bev_grid_size, 
            grid_cell_width, vertical_sampling_range,n_samples))
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.norm = nn.GroupNorm(16, out_ch)

    def forward(self, features, cam):
        assert len(features) == len(self.scales)        
        
        cams = [torch.stack([cam[:, 0, :] / scale, cam[:, 1, :] /scale, cam[:,2, :]], dim=1) for scale in self.scales]
        
        projections = []
        for f, c in zip(features, cams):
            B, C, H, W = f.shape
            proj_points = torch.einsum('bij, klmj ->bklmi', c, self.volume_grid)
            proj_points = proj_points[:,:, :, :, :2]
            norm_points = proj_points - torch.tensor([W/2, H/2], device=proj_points.device).float()
            norm_points = torch.div(norm_points, torch.tensor([W/2, H/2], device=proj_points.device).float())
            B, X, Y, D, V = norm_points.shape
            points = norm_points.permute((0, 3, 1, 2, 4)).contiguous().view(B, D * X, Y, V)
            sampled_values = F.grid_sample(f, points, padding_mode='zeros', align_corners=True) 
            sampled_values = sampled_values.view(B, C, D, X, Y).contiguous()
            projections.append(sampled_values)

        f_3d = torch.stack(projections, dim=0).mean(dim=0)
        f_3d = F.relu(self.norm(self.conv(f_3d)))
        return f_3d

class ContextNetwork(nn.Module):

    def __init__(self, n_classes, in_ch=128, out_ch=512):
        super().__init__()
        self.unet = UNet3D(in_ch, in_ch, final_sigmoid = False, f_maps = 128, num_groups = 16,
                num_levels=3, is_segmentation=False)
        
        self.conv_out = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1)
        self.norm_out = nn.GroupNorm(16, out_ch)
        #self.conv_collapse = nn.Conv2d(in_ch, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.unet(x)
        out = F.relu(self.norm_out(self.conv_out(x)))
        # x.shape == B, C, D, X, Y
        #collapsed = torch.mean(x, 4)
        #collapsed = self.conv_collapse(collapsed)
        
        return out

class Net3D(nn.Module):

    def __init__(self, scales = [4, 8, 16, 32, 64], in_ch= 256, out_ch = 512, n_classes = 11, 
            bev_grid_size=(49, 50), grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=50, with_context=False):
        super().__init__()        
        self.with_context = with_context

        if self.with_context:
            int_ch = in_ch // 2
            self.ray_projection_layer = RayProjectionLayer(scales, in_ch, int_ch, bev_grid_size,
                grid_cell_width, vertical_sampling_range, n_samples)
            self.context_layer = ContextNetwork(n_classes, int_ch, out_ch)
        else:
            self.ray_projection_layer = RayProjectionLayer(scales, in_ch, out_ch, bev_grid_size,
                grid_cell_width, vertical_sampling_range, n_samples)


    def forward(self, f, cam):
        x = self.ray_projection_layer(f, cam)
        if self.with_context:
            x = self.context_layer(x)
        return x

def _test():
    f = [torch.rand((1, 256, 256 // scale, 1024 // scale)) for scale in [4, 8, 16, 32, 64]]
    cam = torch.tensor([[700, 0, 512], [0, 700, 128], [0, 0,1]], dtype = torch.float32)
    net = Net3D()
    out, collapsed = net(f, cam)
    print(out.shape)
    print(collapsed.shape)

if __name__ == '__main__':
    _test()
