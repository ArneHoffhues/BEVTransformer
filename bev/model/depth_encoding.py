import torch
from torch import nn
import torch.nn.functional as F

from bev.model.utils import construct_volume_lattice
from bev.model.resnet import ResNetLayer

class DepthEncoder(nn.Module):

    def __init__(self, out_ch = 128, bev_grid_size=(50, 50), grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=20, max_depth=80.):
        super().__init__()
        self.grid_cell_width = grid_cell_width
        self.bev_grid_size = bev_grid_size
        self.n_samples = n_samples
        self.max_depth = max_depth

        self.register_buffer('volume_grid', construct_volume_lattice(bev_grid_size, 
            grid_cell_width, vertical_sampling_range,n_samples))
        self.embedding = nn.Linear(5, out_ch)

    def forward(self, depth, cam):      
        
        B, _, H, W = depth.shape
        proj_points = torch.einsum('bij, klmj ->bklmi', cam, self.volume_grid)
        proj_points = proj_points[:,:, :, :, :2]
        norm_points = proj_points - torch.tensor([W/2, H/2], device=proj_points.device).float()
        norm_points = torch.div(norm_points, torch.tensor([W/2, H/2], device=proj_points.device).float())
        B, X, Y, D, V = norm_points.shape
        points = norm_points.permute((0, 3, 1, 2, 4)).contiguous().view(B, D * X, Y, V)
        sampled_depth = F.grid_sample(depth, points, padding_mode='zeros', align_corners=True) 
        sampled_depth = sampled_depth.view(B, 1, D, X, Y).contiguous()

        depth_locations = torch.linspace(1+ (self.bev_grid_size[0] - 0.5) * self.grid_cell_width, 
                1 +self.grid_cell_width/2, self.bev_grid_size[0], device=depth.device)
        depth_locations = depth_locations[None, None, :, None, None].repeat(B, 1, 1, X, Y) / self.max_depth
        depth_difference = torch.abs(sampled_depth - depth_locations)
        depth_difference[sampled_depth == 0] = 0.

        z_coord = torch.arange(self.bev_grid_size[0], device=depth.device)
        z_coord = z_coord[None, None, :, None, None].repeat(B, 1, 1, X, Y)

        x_coord = torch.arange(self.bev_grid_size[1], device=depth.device)
        x_coord = x_coord[None, None, None, :, None].repeat(B, 1, D, 1, Y)

        y_coord = torch.arange(self.n_samples, device=depth.device)
        y_coord = y_coord[None, None, None, None, :].repeat(B, 1, D, X, 1)

        embed_info = torch.cat([sampled_depth, depth_difference, z_coord, x_coord, y_coord], dim = 1)
        embed_info = embed_info.permute(0, 2, 3, 4, 1)
        embedded = self.embedding(embed_info)
        embedded = embedded.permute(0, 4, 1, 2, 3)
        return embedded

class SimpleDepthEncoder(nn.Module):

    def __init__(self, out_ch = 128, bev_grid_size=(50, 50), grid_cell_width=1, vertical_sampling_range=[-3,3], n_samples=20, max_depth=80.):
        super().__init__()
        self.grid_cell_width = grid_cell_width
        self.bev_grid_size = bev_grid_size
        self.n_samples = n_samples
        self.max_depth = max_depth

        self.register_buffer('volume_grid', construct_volume_lattice(bev_grid_size,
            grid_cell_width, vertical_sampling_range,n_samples))
        self.embedding = nn.Linear(1, out_ch)

    def forward(self, depth, cam):

        B, _, H, W = depth.shape
        proj_points = torch.einsum('bij, klmj ->bklmi', cam, self.volume_grid)
        proj_points = proj_points[:,:, :, :, :2]
        norm_points = proj_points - torch.tensor([W/2, H/2], device=proj_points.device).float()
        norm_points = torch.div(norm_points, torch.tensor([W/2, H/2], device=proj_points.device).float())
        B, X, Y, D, V = norm_points.shape
        points = norm_points.permute((0, 3, 1, 2, 4)).contiguous().view(B, D * X, Y, V)
        sampled_depth = F.grid_sample(depth, points, padding_mode='zeros', align_corners=True)
        sampled_depth = sampled_depth.view(B, 1, D, X, Y).contiguous()
        
        sampled_depth = sampled_depth.permute(0, 2, 3, 4, 1)
        embedded = self.embedding(sampled_depth)
        embedded = embedded.permute(0, 4, 1, 2, 3)

        return embedded

def _test():
    f = [torch.rand((1, 256, 256 // scale, 1024 // scale)) for scale in [4, 8, 16, 32, 64]]
    cam = torch.tensor([[700, 0, 512], [0, 700, 128], [0, 0,1]], dtype = torch.float32)
    net = Net3D()
    out, collapsed = net(f, cam)
    print(out.shape)
    print(collapsed.shape)

if __name__ == '__main__':
    _test()
