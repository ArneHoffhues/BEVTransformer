import torch
from torch import nn
import torch.nn.functional as F

from bev.model.utils import construct_volume_lattice
from bev.model.resnet import ResNetLayer
from bev.model.unet.unet_model import UNet3D
from bev.model.depth_encoding import DepthEncoder, SimpleDepthEncoder

#from utils import construct_volume_lattice
#from resnet import ResNetLayer
#from unet.unet_model import UNet3D
#from depth_encoding import DepthEncoder, SimpleDepthEncoder


class ProjectionLayer(nn.Module):

    def __init__(self, scales = [4, 8, 16, 32, 64], in_ch= 256, out_ch = 128, bev_grid_size=(50, 50), 
            grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=20, with_depth=False, with_depth_encoding=False, ground_projected=False):
        super().__init__()
        self.scales = scales
        self.with_depth = with_depth
        self.with_depth_encoding= with_depth_encoding
        self.ground_projected = ground_projected

        self.register_buffer('volume_grid', construct_volume_lattice(bev_grid_size, 
            grid_cell_width, vertical_sampling_range,n_samples))
        if self.with_depth:
            self.reduce_conv = nn.Conv3d(in_ch, in_ch-1, kernel_size = 1, stride = 1)
        if self.with_depth_encoding:
            self.depth_encoder = SimpleDepthEncoder(out_ch = out_ch, bev_grid_size=bev_grid_size, grid_cell_width=grid_cell_width, 
                    vertical_sampling_range = vertical_sampling_range, n_samples = n_samples)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size = 1, stride = 1)
        self.norm = nn.GroupNorm(16, out_ch)

    def forward(self, features, cam, depth=None):
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

        if self.with_depth:
            assert depth is not None
            sampled_depth = F.grid_sample(depth, points, padding_mode='zeros', align_corners=True)
            sampled_depth = sampled_depth.view(B, 1, D, X, Y).contiguous()
            f_3d = self.reduce_conv(f_3d)
            f_3d = torch.cat([f_3d, sampled_depth], dim=1)

        if self.with_depth_encoding:
            assert depth is not None
            encoded_depth = self.depth_encoder(depth, cam)
            f_3d = f_3d + encoded_depth

        f_3d = F.relu(self.norm(self.conv(f_3d)))

        if self.ground_projected:
            f_3d = torch.mean(f_3d, dim=-1)
        return f_3d

class ContextNetwork(nn.Module):

    def __init__(self, n_classes, in_ch=128, out_ch=256):
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

    def __init__(self, scales = [4, 8, 16, 32, 64], in_ch= 256, out_ch = 256, n_classes = 10, 
            bev_grid_size=(50, 50), grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=20, initialization_level = -0.5,
            with_context=False, with_depth=False, with_depth_encoding=False, ground_projected=False):
        super().__init__()
        if ground_projected and (with_context or with_depth or with_depth_encoding):
            raise ValueError('ground_projected and with_context, with_depth and with_depth_encoding are mutually exclusive!')
        
        self.ground_projected = ground_projected
        self.with_context = with_context
        self.initialization_level = initialization_level
        self.vertical_sampling_range = vertical_sampling_range
        self.n_samples = n_samples

        if self.with_context:
            int_ch = in_ch // 2
            self.projection_layer = ProjectionLayer(scales, in_ch, int_ch, bev_grid_size,
                grid_cell_width, vertical_sampling_range, n_samples, with_depth=with_depth, with_depth_encoding=with_depth_encoding, ground_projected=ground_projected)
            self.context_layer = ContextNetwork(n_classes, int_ch, out_ch)
        else:
            self.projection_layer = ProjectionLayer(scales, in_ch, out_ch, bev_grid_size,
                grid_cell_width, vertical_sampling_range, n_samples, with_depth=with_depth, with_depth_encoding=with_depth_encoding, ground_projected=ground_projected)

    def forward(self, f, cam, depth=None):
        x = self.projection_layer(f, cam, depth)
        if self.with_context:
            x = self.context_layer(x)
        
        if not self.ground_projected:
            B, C, D, X, Y = x.shape
            #init_level = torch.tensor([self.config.initialization_level], dtype=torch.float32)
            init_level = self.initialization_level
            #vert_samples = torch.linspace(self.config.vertical_sampling_range[0], self.config.vertical_sampling_range[1], self.config.n_samples_vertical)
            vert_samples = torch.linspace(self.vertical_sampling_range[0], self.vertical_sampling_range[1], self.n_samples, device=x.device)
            assert init_level >= vert_samples[0] and init_level < vert_samples[-1]

            init_index = torch.searchsorted(vert_samples, init_level)

            init= x[:, :, :, :, init_index].clone().detach()
        else:
            init = x
            x = None

        return x, init

class TransformModule(nn.Module):
    def __init__(self, dim=(6, 22), out=(50, 50)):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.out = out

        self.fc_transform = nn.Sequential(
                        nn.Linear(dim[0] * dim[1], dim[0] * dim[1]),
                        nn.ReLU(),
                        nn.Linear(dim[0] * dim[1], out[0] * out[1]),
                        nn.ReLU()
                    )

    def forward(self, f, cam):
        x = f[-1]
        x = x.view(list(x.size()[:2]) + [self.dim[0] * self.dim[1],])
        x = self.fc_transform(x)
        init = x.view(list(x.size()[:2]) + list(self.out))
        return None, init

def _test():
    f = [torch.rand((1, 256, 256 // scale, 1024 // scale)) for scale in [4, 8, 16, 32, 64]]
    cam = torch.tensor([[700, 0, 512], [0, 700, 128], [0, 0,1]], dtype = torch.float32)
    cam = cam[None, :, :]
    #net = Net3D()
    net = TransformModule()
    x, init = net(f, cam)
    print(x)
    print(init.shape)

if __name__ == '__main__':
    _test()
